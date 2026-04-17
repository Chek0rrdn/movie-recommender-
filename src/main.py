#!/usr/bin/env python3
"""
Sistema de Recomendación de Películas
Basado en filtrado por contenido y similitud de coseno.

Uso:
    python3 main.py ratings.csv
    python3 main.py ratings.csv --top 30

Requiere archivo .env con:
    TMDB_API_KEY=tu_api_key
"""

import sys
import os
import argparse
import time
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================
# 1. CONFIGURACIÓN
# ============================
def cargar_config():
    """Carga la API key desde .env"""
    # Busca .env en el directorio actual o en el padre
    if os.path.exists('.env'):
        load_dotenv('.env')
    elif os.path.exists('../.env'):
        load_dotenv('../.env')
    else:
        print("Error: No se encontró archivo .env")
        print("Crea uno con: echo 'TMDB_API_KEY=tu_key' > .env")
        sys.exit(1)

    api_key = os.getenv('TMDB_API_KEY')
    if not api_key:
        print("Error: TMDB_API_KEY no encontrada en .env")
        sys.exit(1)

    return api_key


# ============================
# 2. CARGA Y LIMPIEZA
# ============================
def cargar_csv(path):
    """Carga y limpia el CSV de Letterboxd"""
    if not os.path.exists(path):
        print(f"Error: No se encontró el archivo '{path}'")
        sys.exit(1)

    df = pd.read_csv(path)

    # Verificar columnas esperadas
    columnas_requeridas = ['Name', 'Rating']
    for col in columnas_requeridas:
        if col not in df.columns:
            print(f"Error: Columna '{col}' no encontrada en el CSV")
            print(f"Columnas disponibles: {df.columns.tolist()}")
            sys.exit(1)

    df = df.rename(columns={
        'Name': 'title',
        'Year': 'year',
        'Rating': 'my_score',
        'Letterboxd URI': 'letterboxd_url',
        'Date': 'date_rated'
    })

    df = df[df['my_score'] > 0].copy()
    df.reset_index(drop=True, inplace=True)

    print(f"Películas cargadas: {len(df)}")
    print(f"Calificación promedio: {df['my_score'].mean():.2f}")
    print(f"Rango: {df['my_score'].min()} - {df['my_score'].max()}")

    return df


# ============================
# 3. OBTENER GÉNEROS DE TMDb
# ============================
def search_movie(title, year, api_key):
    """Busca una película en TMDb y devuelve su ID"""
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {'api_key': api_key, 'query': title, 'year': year}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                return results[0]['id']
        return None
    except Exception:
        return None


def get_genres(tmdb_id, api_key):
    """Obtiene los géneros de una película por su ID de TMDb"""
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {'api_key': api_key}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [g['name'] for g in data.get('genres', [])]
        return []
    except Exception:
        return []


def obtener_todos_los_generos(df, api_key):
    """Obtiene géneros de todas las películas del dataset"""
    print("\nObteniendo géneros desde TMDb...")
    tmdb_ids = {}
    genres_dict = {}
    total = len(df)

    for i, row in df.iterrows():
        title = row['title']
        year = row.get('year', '')

        tmdb_id = search_movie(title, year, api_key)
        time.sleep(0.25)

        if tmdb_id:
            tmdb_ids[i] = tmdb_id
            genres = get_genres(tmdb_id, api_key)
            time.sleep(0.25)
            genres_dict[i] = genres
        else:
            genres_dict[i] = []

        if (i + 1) % 10 == 0 or (i + 1) == total:
            pct = ((i + 1) / total) * 100
            print(f"  Progreso: {i + 1}/{total} ({pct:.0f}%) - {title}")

    df['genres'] = df.index.map(genres_dict)
    df['genres_str'] = df['genres'].apply(lambda x: ', '.join(x) if x else 'Unknown')

    # Filtrar películas sin géneros
    antes = len(df)
    df = df[df['genres'].apply(len) > 0].copy()
    df.reset_index(drop=True, inplace=True)

    found = len(df)
    print(f"\nGéneros encontrados para {found}/{antes} películas ({found/antes*100:.1f}%)")

    return df, tmdb_ids


# ============================
# 4. CONSTRUIR PERFIL Y MODELO
# ============================
def construir_perfil(df):
    """Construye el perfil de usuario ponderado por calificaciones"""
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genres'])

    weights = df['my_score'].values.reshape(-1, 1)
    weighted_genres = genre_matrix * weights
    user_profile = weighted_genres.sum(axis=0) / weights.sum()

    perfil = pd.Series(user_profile, index=mlb.classes_).sort_values(ascending=False)

    print("\n" + "=" * 50)
    print("TU PERFIL DE GUSTOS")
    print("=" * 50)
    for genre, weight in perfil.items():
        bar = "█" * int(weight * 40)
        print(f"  {genre:<20} {weight:.3f} {bar}")

    return mlb, user_profile


# ============================
# 5. BUSCAR RECOMENDACIONES
# ============================
def buscar_recomendaciones(df, mlb, user_profile, tmdb_ids, api_key, top_n=20):
    """Busca películas nuevas y calcula similitud con el perfil"""
    print("\nBuscando películas candidatas...")

    # Obtener IDs de géneros de TMDb
    response = requests.get(
        "https://api.themoviedb.org/3/genre/movie/list",
        params={'api_key': api_key},
        timeout=10
    )
    genre_ids = {g['name']: g['id'] for g in response.json()['genres']}

    # Top 5 géneros del usuario
    perfil = pd.Series(user_profile, index=mlb.classes_).sort_values(ascending=False)
    top_genres = perfil.head(5).index.tolist()
    print(f"  Buscando por géneros: {', '.join(top_genres)}")

    # Buscar candidatos
    my_tmdb_ids = set(tmdb_ids.values())
    nuevas = []

    for genre_name in top_genres:
        if genre_name not in genre_ids:
            continue
        gid = genre_ids[genre_name]

        for page in [1, 2]:
            url = "https://api.themoviedb.org/3/discover/movie"
            params = {
                'api_key': api_key,
                'with_genres': gid,
                'sort_by': 'vote_average.desc',
                'vote_count.gte': 500,
                'page': page
            }
            try:
                response = requests.get(url, params=params, timeout=10)
                time.sleep(0.25)
                if response.status_code == 200:
                    for movie in response.json().get('results', []):
                        if movie['id'] not in my_tmdb_ids:
                            genres = get_genres(movie['id'], api_key)
                            time.sleep(0.25)
                            nuevas.append({
                                'tmdb_id': movie['id'],
                                'title': movie['title'],
                                'year': movie.get('release_date', '')[:4],
                                'score_tmdb': movie.get('vote_average', 0),
                                'genres': genres
                            })
            except Exception:
                continue

    df_nuevas = pd.DataFrame(nuevas).drop_duplicates(subset='tmdb_id')
    print(f"  Películas candidatas encontradas: {len(df_nuevas)}")

    if len(df_nuevas) == 0:
        print("No se encontraron candidatas. Intenta con más películas calificadas.")
        return pd.DataFrame()

    # Calcular similitud
    nuevas_matrix = mlb.transform(df_nuevas['genres'])
    similitudes = cosine_similarity([user_profile], nuevas_matrix)[0]

    df_nuevas['similitud'] = similitudes
    df_nuevas['genres_str'] = df_nuevas['genres'].apply(lambda x: ', '.join(x))

    recomendaciones = df_nuevas.sort_values('similitud', ascending=False).head(top_n)
    return recomendaciones


# ============================
# 6. MOSTRAR RESULTADOS
# ============================
def mostrar_resultados(recomendaciones):
    """Muestra las recomendaciones de forma formateada"""
    print("\n" + "=" * 70)
    print(f"  TOP {len(recomendaciones)} RECOMENDACIONES PARA TI")
    print("=" * 70)

    for i, (_, row) in enumerate(recomendaciones.iterrows(), 1):
        sim_pct = row['similitud'] * 100
        print(f"\n  {i:>2}. {row['title']} ({row['year']})")
        print(f"      Score TMDb: {row['score_tmdb']:.1f}  |  Similitud: {sim_pct:.0f}%")
        print(f"      Géneros: {row['genres_str']}")

    print("\n" + "=" * 70)


# ============================
# EJECUCIÓN PRINCIPAL
# ============================
def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendación de Películas basado en similitud de coseno.",
        epilog="Ejemplo: python3 main.py ratings.csv --top 15"
    )
    parser.add_argument(
        'csv_path',
        help="Ruta al archivo ratings.csv exportado de Letterboxd"
    )
    parser.add_argument(
        '--top', '-n',
        type=int,
        default=20,
        help="Número de recomendaciones a generar (default: 20)"
    )
    parser.add_argument(
        '--guardar', '-g',
        type=str,
        default=None,
        help="Ruta para guardar las recomendaciones en CSV (opcional)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  SISTEMA DE RECOMENDACIÓN DE PELÍCULAS")
    print("  Basado en filtrado por contenido y similitud de coseno")
    print("=" * 70)

    # 1. Configuración
    api_key = cargar_config()
    print("✓ API Key cargada")

    # 2. Cargar datos
    df = cargar_csv(args.csv_path)

    # 3. Obtener géneros
    df, tmdb_ids = obtener_todos_los_generos(df, api_key)

    # 4. Construir perfil
    mlb, user_profile = construir_perfil(df)

    # 5. Buscar recomendaciones
    recomendaciones = buscar_recomendaciones(df, mlb, user_profile, tmdb_ids, api_key, args.top)

    if recomendaciones.empty:
        return

    # 6. Mostrar resultados
    mostrar_resultados(recomendaciones)

    # 7. Guardar si se solicitó
    if args.guardar:
        recomendaciones[['title', 'year', 'score_tmdb', 'similitud', 'genres_str']].to_csv(
            args.guardar, index=False
        )
        print(f"\n✓ Recomendaciones guardadas en: {args.guardar}")


if __name__ == '__main__':
    main()