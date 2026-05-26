# Machine Learning con PySpark y Docker — Proyecto Final

**Autor:** Sergio Prieto  
**Curso:** Machine Learning con PySpark y Docker  
**Profesora:** Luz Adriana Gutiérrez Rodríguez  
**Universidad Santo Tomás — Programa de Estadística**  
**Semestre 2026-I | Mayo 2026**

---

## 1. Descripción del Proyecto

Proyecto final del curso Machine Learning con PySpark y Docker. Consiste en un análisis 360° sobre datos reales colombianos, aplicando todo el contenido del semestre: desde análisis exploratorio con PySpark hasta modelos pre-entrenados de inteligencia artificial.

El proyecto se estructura en tres bloques integrados sobre el mismo fenómeno de estudio: la vigilancia en salud pública en Colombia.

---

## 2. Dataset

### Dataset principal — SIVIGILA 2019
- **Fuente:** Instituto Nacional de Salud de Colombia — Sistema Nacional de Vigilancia en Salud Pública
- **Registros:** 205,532 notificaciones obligatorias
- **Variables:** Código del evento, nombre del evento, año, semana epidemiológica, código de departamento, municipio, conteo de casos
- **Eventos:** 69 tipos de eventos de salud pública en 35 departamentos y 52 semanas

### Corpus textual — PQRS datos.gov.co
- **Fuente:** Portal de Datos Abiertos del Estado Colombiano (datos.gov.co, id: `e88e-ctba`)
- **Documentos:** 616 peticiones, quejas, reclamos y sugerencias reales de ciudadanos
- **Variables:** Identificador PQRS, asunto (texto libre), categoría administrativa, fecha
- **Uso:** Bloque 3 — NLP (tokenización, TF-IDF, clasificación, análisis de sentimiento)

---

## 3. Metodología

### Bloque 1 — Análisis Exploratorio con PySpark
- Carga del dataset con `SparkSession` y limpieza de tipos
- 3 transformaciones analíticas: `filter` (eventos DENGUE), `groupBy` (total por evento), `withColumn` + `join` (tasa de incidencia)
- Estadística descriptiva: `describe()`, `summary()`, percentiles, detección de nulos, duplicados y valores atípicos (método IQR)
- 2 visualizaciones: top eventos por casos y serie temporal de dengue
- 3 conclusiones cuantitativas redactadas

### Bloque 2 — Machine Learning con Spark ML
- **Pipeline:** `StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `StandardScaler`
- **No supervisado:** PCA (32 componentes, 92.6% varianza) + K-Means (método del codo, K=10 clusters)
- **Supervisado:** Regresión Logística + Random Forest (7 clases de eventos)
- **Métricas:** Accuracy, Precision, Recall, F1, AUC (One-vs-Rest)
- **Validación cruzada:** `CrossValidator` con 3 folds, grid de `maxDepth` y `numTrees`

### Bloque 3 — NLP y Modelo Pre-entrenado
- **Corpus real:** 616 PQRS ciudadanas de datos.gov.co tokenizadas con `RegexTokenizer` + 119 stop words en español
- **TF-IDF:** `CountVectorizer` + `IDF`, comparación TF crudo vs TF-IDF por categoría
- **Clasificación:** Regresión Logística sobre vectores TF-IDF, matriz de confusión, coeficientes interpretables
- **Hugging Face:** Modelo `pysentimiento/robertuito-sentiment-analysis` aplicado al corpus completo
- **Comparación:** 5 casos difíciles probados con ambos enfoques

---

## 4. Resultados

| Bloque | Métrica principal | Resultado |
|--------|------------------|-----------|
| 1 — EDA | Top 3 eventos | Agresiones rabia (141K), Dengue (123K), VCM/VIF (118K) — 49% del total |
| 1 — EDA | Pico dengue | Semana 27 con 3,559 casos |
| 2 — ML | Accuracy RF | 29.1% (vs 14% aleatorio, 7 clases) |
| 2 — ML | AUC (OvR weighted) | Reportado para LR y RF |
| 2 — ML | Feature importance | Departamento 68%, conteo_casos 31%, SEMANA 0.3% |
| 3 — NLP | Corpus PQRS | 616 documentos reales, 9 categorías, TTR moderado |
| 3 — NLP | TF-IDF + LR accuracy | ~45-65% (clasificación realista sobre texto ciudadano) |
| 3 — NLP | Hugging Face | Sentimiento mixto (POS/NEG/NEU) detectado en PQRS reales |

### Conclusiones principales
1. **Concentración de morbilidad:** 3 patologías concentran ~49% de las notificaciones SIVIGILA
2. **Estacionalidad:** El dengue alcanza su pico en semanas 20-35, coincidiendo con temporada de lluvias
3. **Concentración geográfica:** 10 departamentos acumulan >60% de los casos notificados
4. **Limitación ML:** Las features temporales y geográficas solas no bastan para predecir categoría del evento (accuracy 29%)
5. **NLP productivo:** Sistema híbrido TF-IDF (clasificar PQRS) + Hugging Face (priorizar por sentimiento) recomendado para producción

---

## 5. Cómo Reproducir

### Requisitos
- Python 3.11+
- PySpark 4.1.1
- Java 17+ (OpenJDK)
- 4 GB RAM disponibles

### Ejecución local

```bash
# Clonar repositorio
git clone <repo-url>
cd proyecto_final

# Instalar dependencias
pip install pyspark pandas matplotlib seaborn transformers torch pysentimiento scikit-learn jupyter

# Ejecutar notebooks
jupyter notebook notebooks/
```

### Ejecución con Docker

```bash
docker-compose up --build
# Abrir http://localhost:8888
```

### Estructura del proyecto

```
proyecto_final/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── data/
│   ├── sivigila.csv
│   └── pqrs_colombia.csv
├── notebooks/
│   ├── bloque1_eda_prieto.ipynb
│   ├── bloque2_ml_prieto.ipynb
│   └── bloque3_nlp_prieto.ipynb
└── salidas/
    ├── fig1_top15_eventos.png
    ├── fig2_dengue_semanal.png
    ├── fig3_pca_varianza.png
    ├── fig4_elbow_kmeans.png
    ├── fig5_matriz_confusion_*.png (x2)
    ├── fig6_feature_importance.png
    ├── fig7_matriz_confusion_tfidf.png
    ├── fig8_corpus_stats.png
    └── fig9_sentimientos_hf.png
```

---

**Nota:** Los notebooks están ejecutados con todas las celdas y salidas visibles. Las figuras se generan automáticamente en `salidas/` al ejecutar cada notebook.
