"""
============================================================
BLOQUE 3 - NLP Y MODELO PRE-ENTRENADO
============================================================
Estrategia: Se construye un corpus textual a partir del dataset
SIVIGILA, generando documentos descriptivos en lenguaje natural
para cada combinacion (evento x departamento). Las categorias
de los eventos de salud publica son la variable objetivo.

Partes:
  A - Preparacion del corpus (tokenizacion, stop words, TTR, hapax)
  B - TF-IDF y analisis estadistico
  C - Clasificacion con TF-IDF (Regresion Logistica)
  D - Modelo pre-entrenado Hugging Face (pysentimiento)
============================================================
Ejecutar desde terminal:
    python bloque3_nlp.py
"""
import os
import sys

# Fix PySpark en Windows
HADOOP_HOME = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "hadoop_tmp")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.makedirs(os.path.join(HADOOP_HOME, "bin"), exist_ok=True)
winutils_path = os.path.join(HADOOP_HOME, "bin", "winutils.exe")
if not os.path.exists(winutils_path):
    with open(winutils_path, "wb") as f:
        f.write(b"")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum as _sum, count, concat_ws, lit, collect_list,
    regexp_replace, trim, when, round as spark_round, size as spark_size,
    avg as spark_avg, min as spark_min, max as spark_max,
)
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, CountVectorizer, IDF,
    StringIndexer, VectorAssembler,
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ============================================================
# CONFIGURACION
# ============================================================
CSV_PATH = r"C:\Users\MAURICIO\Downloads\data\sivigila.csv"
OUTPUT_DIR = r"C:\Users\MAURICIO\Downloads\data\salidas_bloque3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

# ============================================================
# CATEGORIAS (mismas que Bloque 2)
# ============================================================
CATEGORIA_MAP = {
    205: "VECTORIAL", 210: "VECTORIAL", 217: "VECTORIAL", 220: "VECTORIAL",
    420: "VECTORIAL", 430: "VECTORIAL", 440: "VECTORIAL",
    460: "VECTORIAL", 470: "VECTORIAL", 490: "VECTORIAL", 495: "VECTORIAL",
    540: "VECTORIAL", 580: "VECTORIAL", 895: "VECTORIAL",
    230: "INMUNOPREVENIBLE", 320: "INMUNOPREVENIBLE",
    330: "INMUNOPREVENIBLE", 340: "INMUNOPREVENIBLE", 341: "INMUNOPREVENIBLE",
    345: "INMUNOPREVENIBLE", 348: "INMUNOPREVENIBLE",
    500: "INMUNOPREVENIBLE", 510: "INMUNOPREVENIBLE",
    520: "INMUNOPREVENIBLE", 530: "INMUNOPREVENIBLE",
    620: "INMUNOPREVENIBLE", 730: "INMUNOPREVENIBLE",
    760: "INMUNOPREVENIBLE", 770: "INMUNOPREVENIBLE", 800: "INMUNOPREVENIBLE",
    810: "INMUNOPREVENIBLE", 820: "INMUNOPREVENIBLE", 825: "INMUNOPREVENIBLE",
    831: "INMUNOPREVENIBLE",
    110: "MATERNO_INFANTIL", 112: "MATERNO_INFANTIL", 113: "MATERNO_INFANTIL",
    298: "MATERNO_INFANTIL", 343: "MATERNO_INFANTIL",
    549: "MATERNO_INFANTIL", 550: "MATERNO_INFANTIL", 560: "MATERNO_INFANTIL",
    590: "MATERNO_INFANTIL", 600: "MATERNO_INFANTIL",
    735: "MATERNO_INFANTIL", 740: "MATERNO_INFANTIL", 750: "MATERNO_INFANTIL",
    100: "ZOONOTICO", 300: "ZOONOTICO", 450: "ZOONOTICO", 455: "ZOONOTICO",
    228: "INTOX_VIOLENCIA", 356: "INTOX_VIOLENCIA",
    360: "INTOX_VIOLENCIA", 370: "INTOX_VIOLENCIA", 380: "INTOX_VIOLENCIA",
    390: "INTOX_VIOLENCIA", 400: "INTOX_VIOLENCIA", 410: "INTOX_VIOLENCIA",
    412: "INTOX_VIOLENCIA", 414: "INTOX_VIOLENCIA",
    452: "INTOX_VIOLENCIA", 875: "INTOX_VIOLENCIA",
    155: "CRONICO", 456: "CRONICO", 457: "CRONICO", 459: "CRONICO",
    850: "CRONICO", 305: "OTROS",
}

CATEGORIA_LABEL_MAP = {
    "VECTORIAL": 0, "INMUNOPREVENIBLE": 1, "MATERNO_INFANTIL": 2,
    "ZOONOTICO": 3, "INTOX_VIOLENCIA": 4, "CRONICO": 5, "OTROS": 6,
}

# ============================================================
# PARTE A: PREPARACION DEL CORPUS (Tareas 21-24)
# ============================================================
print("=" * 60)
print("PARTE A: Preparacion del corpus textual")
print("=" * 60)

spark = (
    SparkSession.builder
    .appName("SIVIGILA_Bloque3_NLP")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# --- Cargar y preparar datos ---
print("\n--- Tarea 21: Carga y generacion del corpus textual ---")
pdf_raw = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8", keep_default_na=False)
pdf_raw = pdf_raw.rename(columns={"ANO": "ANO_STR"})

# Mapear categorias
cod_to_cat = {str(k): v for k, v in CATEGORIA_MAP.items()}
cod_to_label = {str(k): CATEGORIA_LABEL_MAP[v] for k, v in CATEGORIA_MAP.items()}
pdf_raw["categoria"] = pdf_raw["COD_EVE"].map(cod_to_cat).fillna("OTROS")
pdf_raw["label"] = pdf_raw["COD_EVE"].map(cod_to_label)
pdf_raw["label"] = pdf_raw["label"].fillna(CATEGORIA_LABEL_MAP["OTROS"])

df = spark.createDataFrame(pdf_raw)
df = (
    df
    .withColumn("conteo_casos", col("conteo_casos").cast(IntegerType()))
    .withColumn("SEMANA", col("SEMANA").cast(IntegerType()))
    .withColumn("COD_DPTO_O", col("COD_DPTO_O").cast(IntegerType()))
    .withColumn("COD_MUN_O", col("COD_MUN_O").cast(IntegerType()))
    .withColumn("COD_EVE", col("COD_EVE").cast(IntegerType()))
    .withColumn("label", col("label").cast(IntegerType()))
    .withColumn("ANO",
        regexp_replace(col("ANO_STR"), r"\.", "").cast(IntegerType()))
    .withColumn("nom_mun",
        when(trim(col("nom_mun")) == "", lit("MUNICIPIO DESCONOCIDO"))
        .otherwise(trim(col("nom_mun"))))
    .drop("ANO_STR")
)

# --- Generar corpus: un documento por cada (Nombre x COD_DPTO_O) ---
# Agrupar por evento y departamento, generar texto descriptivo
print("  Generando documentos del corpus (evento x departamento)...")
df_agg = (
    df.groupBy("Nombre", "COD_DPTO_O", "categoria", "label")
    .agg(
        _sum("conteo_casos").alias("total_casos"),
        count("*").alias("num_registros"),
        collect_list("SEMANA").alias("semanas_list"),
        collect_list("nom_mun").alias("municipios_list"),
    )
)

# Crear documento textual por cada grupo (SIN incluir la categoria explicitamente)
df_agg = df_agg.withColumn(
    "documento",
    concat_ws(" ",
        lit("Reporte de vigilancia epidemiologica. Evento de salud publica:"),
        col("Nombre"),
        lit("en el departamento codigo"),
        col("COD_DPTO_O").cast("string"),
        lit(". Total de casos notificados:"),
        col("total_casos").cast("string"),
        lit(". Numero de registros:"),
        col("num_registros").cast("string"),
        lit(". Este evento de notificacion obligatoria hace parte del sistema"
             " nacional de vigilancia en salud publica SIVIGILA. Los casos"
             " fueron reportados a lo largo del ano en diferentes semanas"
             " epidemiologicas y municipios del departamento correspondiente."
             " La notificacion se realiza de acuerdo con los protocolos"
             " establecidos por el Instituto Nacional de Salud de Colombia."),
    )
)

total_docs = df_agg.count()
print(f"  Corpus generado: {total_docs} documentos")
print(f"  Distribucion por categoria:")
df_agg.groupBy("categoria", "label").count().orderBy("label").show(10, truncate=False)

# --- Tarea 22: Tokenizacion con RegexTokenizer y stop words ---
print("\n--- Tarea 22: Tokenizacion y filtrado de stop words ---")

# Stop words en espanol
STOP_WORDS_ES = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
    "por", "un", "para", "con", "no", "una", "su", "al", "es", "lo",
    "como", "mas", "pero", "sus", "le", "ya", "o", "fue", "este",
    "ha", "entre", "cuando", "todo", "esta", "ser", "son", "tambien",
    "fue", "habia", "era", "muy", "anos", "desde", "hasta", "donde",
    "solo", "durante", "cada", "e", "i", "u", "sobre", "sin", "tiene",
    "han", "otros", "porque", "todos", "cual", "vez", "otro", "tanto",
    "despues", "antes", "si", "puede", "parte", "hace", "dia", "forma",
    "tipo", "gran", "caso", "tener", "bien", "mayor", "alguna", "asi",
    "luego", "dentro", "aunque", "hecho", "sido", "tres", "hacer",
    "mismo", "debido", "cuenta", "estos", "pueden", "ellas", "general",
    "menos", "diferentes", "mejor", "ademas", "casi", "veces",
    "nuestro", "tambien", "tan", "tras", "toda", "siendo", "dos",
    "misma", "ningun", "dicho", "fuera", "siempre", "largo", "mucho",
    "poco", "medio", "nueva", "gran", "buen", "ninguna",
]
print(f"  Stop words cargadas: {len(STOP_WORDS_ES)}")

tokenizer = RegexTokenizer(
    inputCol="documento", outputCol="tokens_raw",
    pattern=r"\s+", minTokenLength=2, toLowercase=True,
)
stop_remover = StopWordsRemover(
    inputCol="tokens_raw", outputCol="tokens",
    stopWords=STOP_WORDS_ES, caseSensitive=False,
)

df_tokenized = tokenizer.transform(df_agg)
df_tokenized = stop_remover.transform(df_tokenized).cache()
_ = df_tokenized.count()

print("\n  Muestra de tokens (3 documentos):")
df_tokenized.select("Nombre", "tokens").show(3, truncate=100)

# --- Tarea 23: Estadisticas del corpus ---
print("\n--- Tarea 23: Estadistica descriptiva del corpus ---")

# Total de tokens por documento (calc en Spark)
df_stats = df_tokenized.withColumn("num_tokens", spark_size("tokens"))

stats_row = df_stats.select(
    count("*").alias("total_docs"),
    _sum("num_tokens").alias("total_tokens"),
    spark_round(spark_avg("num_tokens"), 1).alias("promedio_tokens"),
    spark_min("num_tokens").alias("min_tokens"),
    spark_max("num_tokens").alias("max_tokens"),
).collect()[0]

total_docs_nlp = stats_row["total_docs"]
total_tokens = stats_row["total_tokens"]
avg_tokens = stats_row["promedio_tokens"]
min_t = stats_row["min_tokens"]
max_t = stats_row["max_tokens"]

print(f"  Total de documentos: {total_docs_nlp}")
print(f"  Total de tokens (post stop-words): {total_tokens:,}")
print(f"  Promedio de tokens por documento: {avg_tokens}")
print(f"  Rango de tokens: [{min_t}, {max_t}]")

# Vocabulario y TTR (Type-Token Ratio)
# Recolectar todos los tokens a Pandas para calculo
all_tokens_list = df_tokenized.select("tokens").rdd.flatMap(lambda x: x[0]).collect()
vocabulario = set(all_tokens_list)
vocab_size = len(vocabulario)
ttr = vocab_size / len(all_tokens_list) if all_tokens_list else 0

print(f"  Vocabulario unico (types): {vocab_size:,}")
print(f"  TTR (Type-Token Ratio): {ttr:.4f}")
print(f"  Interpretacion: TTR de {ttr:.4f} indica una riqueza lexica "
      f"{'alta' if ttr > 0.5 else 'moderada' if ttr > 0.3 else 'baja'}, "
      f"tipica de textos tecnicos repetitivos con vocabulario especializado.")

# Top 20 tokens mas frecuentes
token_counts = Counter(all_tokens_list)
print("\n  Top 20 tokens mas frecuentes:")
for token, cnt in token_counts.most_common(20):
    print(f"    {token}: {cnt:,}")

# --- Tarea 24: Hapax legomena ---
print("\n--- Tarea 24: Hapax legomena y reflexion ---")
hapax = [(t, c) for t, c in token_counts.items() if c == 1]
print(f"  Hapax legomena (tokens que aparecen 1 sola vez): {len(hapax):,}")
print(f"  Porcentaje del vocabulario: {len(hapax)/vocab_size*100:.1f}%")
print(f"  Ejemplos: {[h[0] for h in hapax[:15]]}")

print("""
  Reflexion sobre filtrar hapax legomena:
  Los hapax constituyen una porcion significativa del vocabulario en este
  corpus tecnico. Dado que los textos son generados a partir de datos
  estructurados con vocabulario medico especializado, los hapax incluyen
  terminos clinicos y nombres propios que pueden ser informativos para
  la clasificacion. Se decide NO filtrarlos porque:
  1. TF-IDF naturalmente les asigna bajo peso si son ruido
  2. Algunos pueden ser terminos discriminativos de categorias especificas
  3. Con 200+ documentos, el ruido de hapax es manejable
""")

# ============================================================
# PARTE B: TF-IDF (Tareas 25-27)
# ============================================================
print("=" * 60)
print("PARTE B: TF-IDF y analisis estadistico")
print("=" * 60)

# --- Tarea 25: CountVectorizer + IDF ---
print("\n--- Tarea 25: Vectorizacion TF-IDF ---")

# Indexar categorias (label ya lo tenemos numericamente)
label_indexer = StringIndexer(
    inputCol="categoria", outputCol="label_idx", handleInvalid="keep"
)
label_model = label_indexer.fit(df_tokenized)
df_lab = label_model.transform(df_tokenized)

# CountVectorizer
cv = CountVectorizer(
    inputCol="tokens", outputCol="raw_features",
    minDF=2, maxDF=0.9, vocabSize=5000,
)
cv_model = cv.fit(df_lab)
vocab = cv_model.vocabulary
print(f"  Vocabulario del CountVectorizer: {len(vocab)} terminos")

# IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(cv_model.transform(df_lab))
df_feat = idf_model.transform(cv_model.transform(df_lab)).cache()
_ = df_feat.count()

# --- Tarea 26: Palabras con mayor TF-IDF por categoria ---
print("\n--- Tarea 26: Palabras con mayor TF-IDF promedio por categoria ---")

# Calcular TF-IDF promedio por categoria
idf_array = idf_model.idf.toArray()

cat_names = {v: k for k, v in CATEGORIA_LABEL_MAP.items()}
for cat_label in range(7):
    cat_name = cat_names.get(cat_label, f"Cat{cat_label}")
    df_cat = df_feat.filter(col("label") == cat_label)
    cat_count = df_cat.count()
    if cat_count == 0:
        print(f"\n  {cat_name}: Sin documentos")
        continue

    # Sumar vectores TF-IDF de la categoria
    # Convertir a array RDD y sumar
    sum_vector = (
        df_cat.select("features")
        .rdd.map(lambda r: r[0].toArray())
        .reduce(lambda a, b: a + b)
    )
    avg_vector = sum_vector / cat_count

    # Top 10 indices con mayor TF-IDF promedio
    top_indices = np.argsort(avg_vector)[-10:][::-1]
    top_words = [(vocab[i], avg_vector[i]) for i in top_indices]

    print(f"\n  {cat_name} ({cat_count} docs):")
    for word, score in top_words:
        print(f"    {word}: {score:.4f}")

# --- Tarea 27: Comparar TF crudo vs TF-IDF ---
print("\n--- Tarea 27: Comparacion TF crudo vs TF-IDF ---")

# Calcular TF crudo promedio (suma de raw_features / num_docs)
raw_sum = (
    df_feat.select("raw_features")
    .rdd.map(lambda r: r[0].toArray())
    .reduce(lambda a, b: a + b)
)
raw_avg = raw_sum / total_docs_nlp

# TF-IDF promedio global
tfidf_sum = (
    df_feat.select("features")
    .rdd.map(lambda r: r[0].toArray())
    .reduce(lambda a, b: a + b)
)
tfidf_avg = tfidf_sum / total_docs_nlp

print("\n  Top 10 TF crudo (promedio global):")
tf_top = np.argsort(raw_avg)[-10:][::-1]
for i in tf_top:
    print(f"    {vocab[i]}: TF={raw_avg[i]:.4f}")

print("\n  Top 10 TF-IDF (promedio global):")
tfidf_top = np.argsort(tfidf_avg)[-10:][::-1]
for i in tfidf_top:
    print(f"    {vocab[i]}: TF-IDF={tfidf_avg[i]:.4f}")

print("""
  Diferencias TF vs TF-IDF:
  - TF favorece palabras frecuentes en TODO el corpus (ej. 'evento',
    'notificacion', 'salud'), que aparecen en casi todos los documentos.
  - TF-IDF penaliza terminos que aparecen en muchos documentos (IDF bajo)
    y premia terminos frecuentes en POCOS documentos, que son mas
    discriminativos para la clasificacion.
  - Los terminos tecnicos y especificos de cada categoria (nombres de
    enfermedades, terminos clinicos) tienden a tener mayor TF-IDF porque
    aparecen concentrados en los documentos de su categoria.
""")

# ============================================================
# PARTE C: CLASIFICACION CON TF-IDF (Tareas 28-33)
# ============================================================
print("=" * 60)
print("PARTE C: Clasificacion con TF-IDF")
print("=" * 60)

# --- Tarea 28 + 29: Variable objetivo (ya definida en label) ---
print("\n--- Tarea 28-29: Variable objetivo y split ---")
print("  Variable objetivo: 'categoria' del evento de salud (7 clases)")
print("  Nota: El corpus si tiene etiquetas (categorias de SIVIGILA)")

# --- Tarea 30: Train/test split ---
train_nlp, test_nlp = (
    df_feat.select("features", "label")
    .randomSplit([0.8, 0.2], seed=42)
)
print(f"  Train: {train_nlp.count():,} | Test: {test_nlp.count():,}")

# --- Tarea 31: Regresion Logistica sobre TF-IDF ---
print("\n--- Tarea 31: Entrenando Regresion Logistica sobre TF-IDF ---")
lr_nlp = LogisticRegression(
    featuresCol="features", labelCol="label",
    maxIter=100, regParam=0.1,
    family="multinomial",
)
lr_nlp_model = lr_nlp.fit(train_nlp)

# --- Tarea 32: Evaluacion ---
print("\n--- Tarea 32: Metricas del modelo TF-IDF ---")
preds_nlp = lr_nlp_model.transform(test_nlp)

evaluators = {
    "Accuracy": MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"),
    "F1-Score": MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"),
    "Precision": MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"),
    "Recall": MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"),
}

for name, ev in evaluators.items():
    val = ev.evaluate(preds_nlp)
    print(f"    {name}: {val:.4f}")

# Matriz de confusion
cm_nlp = (
    preds_nlp.groupBy("label", "prediction").count()
    .orderBy("label", "prediction").toPandas()
)

n_clases = 7
cm = np.zeros((n_clases, n_clases), dtype=int)
for _, row in cm_nlp.iterrows():
    cm[int(row["label"])][int(row["prediction"])] = int(row["count"])

cat_names_short = {v: k[:12] for k, v in CATEGORIA_LABEL_MAP.items()}
labels_nlp = [cat_names_short.get(i, f"Clase{i}") for i in range(n_clases)]

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_nlp, yticklabels=labels_nlp, ax=ax,
            cbar_kws={"label": "Conteo"})
ax.set_xlabel("Prediccion", fontsize=11)
ax.set_ylabel("Real", fontsize=11)
ax.set_title("Matriz de Confusion - Regresion Logistica (TF-IDF)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig7_matriz_confusion_tfidf.png"), bbox_inches="tight")
plt.close(fig)
print("    Matriz guardada: fig7_matriz_confusion_tfidf.png")

# --- Tarea 33: Coeficientes del modelo ---
print("\n--- Tarea 33: Palabras predictoras por clase (coeficientes LR) ---")
coeff_matrix = lr_nlp_model.coefficientMatrix.toArray()
# coeff_matrix shape: (num_clases, vocab_size)

for cat_label in range(coeff_matrix.shape[0]):
    cat_name = cat_names.get(cat_label, f"Clase{cat_label}")
    row = coeff_matrix[cat_label]
    top_pos = np.argsort(row)[-5:][::-1]  # top 5 coeficientes positivos
    top_neg = np.argsort(row)[:5]         # top 5 coeficientes negativos

    print(f"\n  Clase {cat_label} ({cat_name}):")
    print("    Palabras que PREDICEN esta clase (+):")
    for i in top_pos:
        if i < len(vocab):
            print(f"      {vocab[i]}: {row[i]:.4f}")
    print("    Palabras que ALEJAN de esta clase (-):")
    for i in top_neg:
        if i < len(vocab):
            print(f"      {vocab[i]}: {row[i]:.4f}")

# ============================================================
# PARTE D: MODELO PRE-ENTRENADO HUGGING FACE (Tareas 34-38)
# ============================================================
print("\n" + "=" * 60)
print("PARTE D: Modelo pre-entrenado de Hugging Face")
print("=" * 60)

# --- Tarea 34: Cargar modelo pre-entrenado ---
print("\n--- Tarea 34: Carga de modelo pre-entrenado ---")
print("  Modelo: pysentimiento/robertuito-sentiment-analysis (recomendado)")

# Recolectar textos del corpus a Pandas
pdf_corpus = df_agg.select("documento", "categoria", "label").toPandas()
print(f"  Textos recolectados: {len(pdf_corpus)}")

# Intentar cargar modelo de Hugging Face
try:
    from transformers import pipeline
    print("  Cargando pipeline de sentimiento...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis",
        tokenizer="pysentimiento/robertuito-sentiment-analysis",
        device=-1,  # CPU
    )
    HF_AVAILABLE = True
    print("  Modelo cargado exitosamente.")
except Exception as e:
    print(f"  No se pudo cargar el modelo: {e}")
    print("  Usando modelo fallback basado en reglas...")
    HF_AVAILABLE = False

# --- Tarea 35: Aplicar modelo a una muestra representativa del corpus ---
print("\n--- Tarea 35: Aplicando modelo a muestra del corpus (300 docs) ---")

sample_size = min(300, len(pdf_corpus))
pdf_sample = pdf_corpus.sample(n=sample_size, random_state=42)
textos = pdf_sample["documento"].tolist()

if HF_AVAILABLE:
    batch_size = 32
    all_results = []

    for i in range(0, len(textos), batch_size):
        batch = textos[i:i + batch_size]
        results = sentiment_pipeline(batch, truncation=True, max_length=128)
        all_results.extend(results)
        if (i // batch_size) % 5 == 0:
            print(f"    Procesados {min(i + batch_size, len(textos))}/{len(textos)}...")

    pdf_sample["hf_label"] = [r["label"] for r in all_results]
    pdf_sample["hf_score"] = [r["score"] for r in all_results]

    sent_dist = pdf_sample["hf_label"].value_counts()
    print(f"\n  Distribucion de sentimientos (Hugging Face, muestra={sample_size}):")
    for label, cnt in sent_dist.items():
        print(f"    {label}: {cnt} ({cnt/sample_size*100:.1f}%)")
else:
    print("  Usando clasificador de sentimiento basado en reglas...")
    pos_words = ["alto", "elevado", "grave", "severo", "critico", "epidemia",
                 "brote", "alerta", "emergencia", "complejo"]
    neg_words = ["bajo", "leve", "control", "descenso", "mejora",
                 "disminucion", "estable", "controlado", "positivo"]

    def classify_sentiment(text):
        text_lower = text.lower()
        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)
        if pos_count > neg_count:
            return "NEG"
        elif neg_count > pos_count:
            return "POS"
        return "NEU"

    pdf_sample["hf_label"] = pdf_sample["documento"].apply(classify_sentiment)
    pdf_sample["hf_score"] = 1.0
    sent_dist = pdf_sample["hf_label"].value_counts()
    print(f"\n  Distribucion de sentimientos (reglas, muestra={sample_size}):")
    for label, cnt in sent_dist.items():
        print(f"    {label}: {cnt} ({cnt/sample_size*100:.1f}%)")

# --- Tarea 36: Comparar predicciones ---
print("\n--- Tarea 36: Comparacion TF-IDF vs Hugging Face ---")

# Predicciones del modelo TF-IDF (sobre todo el corpus)
tfidf_preds = lr_nlp_model.transform(df_feat.select("features", "label"))
pdf_corpus["tfidf_pred"] = (
    tfidf_preds.select("prediction").toPandas()["prediction"].values
)

# Mapear sentimiento a categoria real para comparar (solo muestra HF)
hf_label_map = {"POS": 0, "NEU": 1, "NEG": 2}
if "hf_label" in pdf_sample.columns:
    pdf_sample["hf_label_num"] = pdf_sample["hf_label"].map(hf_label_map).fillna(1)
    # Hacer merge con tfidf_pred para la muestra
    pdf_sample = pdf_sample.merge(
        pdf_corpus[["tfidf_pred"]].reset_index(),
        left_index=True, right_index=True, how="left"
    )

print("""
  Nota: Los modelos TF-IDF y Hugging Face predicen variables diferentes:
  - TF-IDF: categoria del evento de salud (7 clases)
  - Hugging Face: sentimiento del texto (POS/NEG/NEU)

  La comparacion directa no es posible. Se evalua si el sentimiento
  esta correlacionado con las categorias reales.
""")

print("  Tabla de contingencia: Categoria real vs Sentimiento (HF, muestra)")
if "hf_label" in pdf_sample.columns:
    contingency = pd.crosstab(pdf_sample["categoria"], pdf_sample["hf_label"])
    print(contingency.to_string())

# --- Tarea 37: Probar 5 casos dificiles ---
print("\n--- Tarea 37: Prueba con 5 casos dificiles ---")

if HF_AVAILABLE:
    casos_dificiles = [
        "El brote de dengue fue completamente controlado y ya no hay casos nuevos.",
        "No se reportaron muertes, lo cual es un excelente resultado para la region.",
        "Aumentaron los casos pero la tasa de mortalidad se mantuvo baja y estable.",
        "La situacion es grave pero no critica, se espera mejoria en las proximas semanas.",
        "A pesar de los esfuerzos de vacunacion, persisten los brotes epidemicos activos.",
    ]

    print("  Evaluando casos con ambos modelos...")
    for i, caso in enumerate(casos_dificiles):
        result_hf = sentiment_pipeline(caso, truncation=True, max_length=128)[0]
        print(f"\n  Caso {i+1}: \"{caso[:80]}...\"")
        print(f"    HF Sentimiento: {result_hf['label']} "
              f"(score: {result_hf['score']:.4f})")

        # TF-IDF no puede predecir textos fuera del vocabulario
        print(f"    TF-IDF: No aplicable (texto fuera del corpus de entrenamiento)")
else:
    print("  Modelo HF no disponible. No se pueden probar casos dificiles.")

# --- Tarea 38: Conclusion sobre que enfoque recomendar ---
print("\n--- Tarea 38: Conclusion y recomendacion final ---")
print("""
  Recomendacion para produccion:
  
  | Aspecto              | TF-IDF + LR          | Hugging Face        |
  |----------------------|----------------------|---------------------|
  | Velocidad            | Muy rapido           | Lento (GPU requerida)|
  | Interpretabilidad    | Alta (coeficientes)  | Baja (caja negra)   |
  | Vocab especializado  | Excelente            | Limitado            |
  | Datos de entreno     | Necesita etiquetas   | Pre-entrenado       |
  | dominio general      | Limitado             | Excelente           |
  
  Para este caso de uso (clasificacion de eventos de salud publica
  colombianos), se recomienda el enfoque TF-IDF + Regresion Logistica
  porque:
  1. El vocabulario medico-tecnico es muy especializado y los modelos
     pre-entrenados no lo capturan bien.
  2. La interpretabilidad es crucial en salud publica (saber QUE palabras
     predicen cada categoria permite auditoria y confianza).
  3. Es mas rapido y no requiere GPU, lo que facilita el despliegue en
     infraestructura del Ministerio de Salud.
  4. Para tareas de sentimiento sobre texto libre (no tecnico), el modelo
     de Hugging Face seria superior.
""")

# ============================================================
# GRAFICOS ADICIONALES DEL BLOQUE 3
# ============================================================
print("\n--- Generando graficos Bloque 3 ---")

# Grafico 7: Distribucion de tokens por documento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

doc_lengths = df_stats.select("num_tokens").toPandas()["num_tokens"]
ax1.hist(doc_lengths, bins=30, color="#2c7bb6", edgecolor="white", alpha=0.8)
ax1.axvline(avg_tokens, color="red", linestyle="--", linewidth=2,
            label=f"Promedio = {avg_tokens}")
ax1.set_xlabel("Tokens por documento")
ax1.set_ylabel("Frecuencia")
ax1.set_title("Distribucion de longitud de documentos")
ax1.legend()

# Grafico 8: Top 30 tokens mas frecuentes
top30 = token_counts.most_common(30)
ax2.barh([t[0] for t in reversed(top30)], [t[1] for t in reversed(top30)],
         color="#2c7bb6")
ax2.set_xlabel("Frecuencia")
ax2.set_title("Top 30 tokens mas frecuentes del corpus")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig8_corpus_stats.png"), bbox_inches="tight")
plt.close(fig)

# Grafico 9: Distribucion de sentimientos (muestra)
fig, ax = plt.subplots(figsize=(7, 4))
sent_colors = {"POS": "#2ca02c", "NEU": "#ff7f0e", "NEG": "#d62728"}
if "hf_label" in pdf_sample.columns:
    sent_dist = pdf_sample["hf_label"].value_counts()
    colors = [sent_colors.get(l, "#7f7f7f") for l in sent_dist.index]
    ax.bar(sent_dist.index, sent_dist.values, color=colors)
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Numero de documentos")
    ax.set_title(f"Distribucion de sentimientos (muestra n={sample_size})")
    for i, (label, val) in enumerate(sent_dist.items()):
        ax.text(i, val + max(sent_dist.values)*0.02, f"{val}\n({val/sample_size*100:.1f}%)",
                ha="center", fontsize=9)
else:
    ax.text(0.5, 0.5, "Datos no disponibles", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_title("Distribucion de sentimientos")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig9_sentimientos_hf.png"), bbox_inches="tight")
plt.close(fig)

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 60)
print("RESUMEN BLOQUE 3 - NLP")
print("=" * 60)
print(f"""
  Corpus generado:        {total_docs_nlp} documentos (evento x departamento)
  Vocabulario unico:      {vocab_size:,} tokens
  TTR (riqueza lexica):   {ttr:.4f}
  Hapax legomena:         {len(hapax):,}

  Modelo TF-IDF + LR:
    Accuracy:  {evaluators['Accuracy'].evaluate(preds_nlp):.4f}
    F1-Score:  {evaluators['F1-Score'].evaluate(preds_nlp):.4f}
    Precision: {evaluators['Precision'].evaluate(preds_nlp):.4f}
    Recall:    {evaluators['Recall'].evaluate(preds_nlp):.4f}

  Modelo Hugging Face:
    Modelo:    pysentimiento/robertuito-sentiment-analysis
    Cargado:   {'Si' if HF_AVAILABLE else 'No (fallback reglas)'}

  Recomendacion: TF-IDF + LR para produccion en salud publica
""")

# Limpiar
df_tokenized.unpersist()
df_feat.unpersist()
spark.stop()
print("Spark detenido. Bloque 3 completado exitosamente.")
print(f"Figuras guardadas en: {OUTPUT_DIR}")
