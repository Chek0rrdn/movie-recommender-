# 🎬 Sistema de Recomendación de Películas

Sistema de recomendación basado en **filtrado por contenido** y **similitud de coseno**, utilizando datos reales de [Letterboxd](https://letterboxd.com/) y la API de [TMDb](https://www.themoviedb.org/).

## ¿Cómo funciona?

1. **Exportas tus calificaciones** de Letterboxd (archivo `ratings.csv`)
2. **Se obtienen los géneros** de cada película mediante la API de TMDb
3. **Se construye tu perfil de gustos** ponderando géneros por tus calificaciones (One-Hot Encoding + promedio ponderado)
4. **Se buscan películas candidatas** que no hayas visto
5. **Se calcula la similitud de coseno** entre tu perfil y cada candidata
6. **Se generan recomendaciones** ordenadas por similitud

## Tecnologías

| Herramienta | Uso |
|---|---|
| Python 3.12 | Lenguaje principal |
| Jupyter Notebook | Desarrollo interactivo |
| pandas | Manipulación de datos |
| NumPy | Operaciones vectoriales |
| scikit-learn | Similitud de coseno, One-Hot Encoding |
| matplotlib / seaborn | Visualizaciones |
| API TMDb | Géneros y metadatos de películas |

## Estructura del proyecto

```
movie-recommender/
├── data/
│   ├── raw/                  # CSV original de Letterboxd
│   └── processed/            # CSVs limpios generados
├── notebooks/
│   ├── 01_carga_y_limpieza.ipynb
│   ├── 02_analisis_exploratorio.ipynb
│   └── 03_modelo_recomendacion.ipynb
├── src/
│   └── main.py               # Script unificado
├── output/                   # Gráficas y recomendaciones
├── .env                      # API key de TMDb (no se sube)
├── .gitignore
├── requirements.txt
└── README.md
```

## Instalación

```bash
git clone git@github.com:Chek0rrdn/movie-recommender-.git
cd movie-recommender-
pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz con tu API key de TMDb:

```
TMDB_API_KEY=tu_api_key_aqui
```

Puedes obtener una API key gratuita en [themoviedb.org](https://www.themoviedb.org/settings/api).

## Uso

### Jupyter Notebooks (paso a paso)

```bash
jupyter notebook
```

Abre los notebooks en orden: `01_carga_y_limpieza` → `02_analisis_exploratorio` → `03_modelo_recomendacion`.

### Script directo

```bash
python src/main.py
```

## Conceptos clave

### One-Hot Encoding

Cada película se convierte en un vector binario de géneros:

```
Spider-Man 2       → [1, 1, 0, 1, 0, 0]  (Action, Sci-Fi, Adventure)
End of Evangelion   → [1, 1, 1, 0, 1, 0]  (Action, Sci-Fi, Animation, Drama)
Your Name           → [0, 0, 1, 0, 1, 1]  (Animation, Drama, Romance)
```

### Perfil ponderado

Cada vector se multiplica por tu calificación. Si le diste 5.0 a Evangelion, ese vector pesa 5x más que una película con 1.0. Así el perfil refleja lo que **disfrutas**, no solo lo que **ves**.

### Similitud de coseno

Mide el ángulo entre tu perfil y cada película candidata. Resultado de 0 a 1:
- **1.0** = idéntico a tus gustos
- **0.0** = nada que ver

## Resultados de ejemplo

| Película | Similitud | Géneros |
|---|---|---|
| My Hero Academia: Heroes Rising | 0.835 | Animation, Action, Fantasy, Adventure |
| Akira | 0.801 | Animation, Sci-Fi, Action |
| Star Wars | 0.798 | Adventure, Action, Sci-Fi |
| Inception | 0.798 | Action, Sci-Fi, Adventure |
| The Lord of the Rings: ROTK | 0.716 | Adventure, Fantasy, Action |

## Licencia

Proyecto académico — uso educativo.
