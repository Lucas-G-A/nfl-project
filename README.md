# Proyecto de NFL – Análisis y Predicción de Partidos

**Autor:** Lucas García y Alonso Zamanillo  
**Repositorio:** https://github.com/Lucas-G-A/nfl-project

Este proyecto analiza el desempeño de equipos de la **NFL** y construye un **modelo de predicción de partidos** utilizando métricas estadísticas y un sistema de ratings **Elo**, sin utilizar machine learning.

## 🎯 Objetivos

- Identificar qué factores explican el éxito de un equipo
- Visualizar estos factores de forma clara mediante gráficas
- Predecir partidos reales y escenarios hipotéticos
- Analizar errores del modelo y su evolución

## 🧠 Metodología

El proyecto se divide en **dos partes principales**:

### 1. Análisis de datos

A partir de datos oficiales de la NFL se construyen métricas como:
- **Diferencial de entregas de balón (turnover margin)**
- **Eficiencia ofensiva** (yardas por jugada)
- **Eficiencia defensiva** (yardas permitidas)
- **Consistencia** en anotación
- **Balance ofensivo–defensivo**
- **Índice de agresividad**

Estas métricas se visualizan en **gráficas** que resumen los hallazgos clave del análisis.

### 2. Predicción de partidos

Se utiliza un sistema **Elo** ajustado por:
- Resultados históricos
- Ventaja de local (home field advantage)
- Métricas de eficiencia calculadas

El modelo genera:
- Probabilidades de victoria para los partidos de la semana
- Predicciones para enfrentamientos hipotéticos
- Análisis contrafactual de escenarios alternativos

## 🛠️ Tecnologías utilizadas

- **Python 3.11**
- **pandas / numpy** – manipulación de datos
- **matplotlib** – visualización
- **nfl-data-py** – datos oficiales de la NFL (schedules y play-by-play)
- **Streamlit** – interfaz interactiva web
- **Docker** – reproducibilidad del entorno

## 📦 Estructura del proyecto

```
nfl-project/
├── notebooks/              # Análisis exploratorio, generación de métricas y predicciones
│   ├── predicciones.ipynb  # Notebook principal para predicciones y Elo
│   ├── graficas.ipynb      # Generación de visualizaciones
│   ├── logos.ipynb         # Manejo de logos de equipos
│   └── NflFunc.ipynb       # Funciones auxiliares
├── figures/                # Gráficas finales exportadas como PNG
├── data/                   # Archivos CSV con predicciones y datos históricos
│   ├── predictions_this_weekend.csv
│   ├── elo_history.csv
│   └── backtest_recent_errors.csv
├── app.py                  # Aplicación Streamlit (interfaz web)
├── elo_ratings.json        # Ratings Elo finales para predicciones
├── latest_team_stats.json  # Estadísticas más recientes de equipos
├── team_logos.json         # URLs/logos de equipos
├── requirements.txt        # Dependencias del proyecto
├── Dockerfile              # Definición del entorno reproducible
└── README.md               # Este archivo
```

## 🚀 Instalación y uso

### Opción 1: Usando Docker (recomendado)

1. Clona el repositorio:
```bash
git clone https://github.com/Lucas-G-A/nfl-project.git
cd nfl-project
```

2. Construye la imagen Docker:
```bash
docker build -t nfl-project .
```

3. Ejecuta el contenedor:
```bash
docker run -p 8501:8501 nfl-project
```

4. Abre tu navegador en `http://localhost:8501`

### Opción 2: Instalación local

1. Clona el repositorio:
```bash
git clone https://github.com/Lucas-G-A/nfl-project.git
cd nfl-project
```

2. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecuta la aplicación Streamlit:
```bash
streamlit run app.py
```

5. Abre tu navegador en `http://localhost:8501`

## 📊 Funcionalidades de la aplicación

La aplicación Streamlit incluye las siguientes secciones:

### 🏠 Inicio
Descripción del proyecto, metodología y tecnologías utilizadas.

### 📈 Análisis
Visualización de las gráficas generadas que muestran las métricas clave:
- Margen de Turnovers
- Índice de agresividad
- Tasa de pase vs eficiencia
- Clasificación ofensiva vs defensiva
- Consistencia vs victorias

### ⚽ Partidos de esta semana
Predicciones para los partidos de la semana actual con probabilidades de victoria calculadas.

### 🎲 Partido hipotético
Permite simular cualquier enfrentamiento entre equipos usando el sistema Elo, con opción de sede neutral.

### 🔄 Contrafactual
Análisis de escenarios alternativos: ¿qué pasa si un equipo mejora ciertas métricas? Permite ajustar:
- Turnover margin
- Eficiencia ofensiva (yardas por jugada)
- Eficiencia defensiva (yardas permitidas)

### ❌ Análisis de errores
Muestra los partidos donde el modelo tuvo mayores errores, incluyendo métricas como:
- Accuracy
- Brier score
- Detalle de errores por partido

### 📉 Historia Elo (por equipo)
Visualización de la evolución del rating Elo de cada equipo a lo largo de la temporada.

## 🔄 Flujo de trabajo

1. **Generación de datos**: Los análisis y predicciones se generan **offline** en los notebooks de Jupyter.
2. **Exportación**: Los resultados se guardan como archivos (`CSV`, `PNG`, `JSON`).
3. **Visualización**: La aplicación Streamlit **consume estos archivos** para mostrarlos de forma interactiva.
4. **Ventajas**: Este enfoque garantiza velocidad, estabilidad y reproducibilidad, replicando sistemas de producción reales donde el cálculo pesado está separado de la capa de presentación.

## 📝 Notas importantes

- El modelo **no utiliza machine learning**; está basado en estadística descriptiva y el sistema Elo.
- Los datos se obtienen de `nfl-data-py`, que utiliza datos oficiales de la NFL.
- Las predicciones se actualizan semanalmente ejecutando los notebooks correspondientes.
- El modelo incorpora ventaja de local (home field advantage) equivalente a ~55 puntos Elo.

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y personal.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request si deseas colaborar.
