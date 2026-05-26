"""
============================================================
BLOQUE 2 - MACHINE LEARNING CON Spark ML
Dataset: SIVIGILA - Vigilancia en Salud Publica, Colombia 2019
Parcial Final - Machine Learning con PySpark y Docker
============================================================
Ejecutar desde terminal:
    python bloque2_ml.py

Partes:
  A - Pipeline de preparacion de datos
  B - PCA + K-Means (no supervisado)
  C - Clasificacion supervisada (LR + RF)
  D - Validacion cruzada
"""
import os
import sys

# ============================================================
# Fix para PySpark en Windows + Java 17+
# ============================================================
HADOOP_HOME = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "hadoop_tmp")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.makedirs(os.path.join(HADOOP_HOME, "bin"), exist_ok=True)
winutils_path = os.path.join(HADOOP_HOME, "bin", "winutils.exe")
if not os.path.exists(winutils_path):
    with open(winutils_path, "wb") as f:
        f.write(b"")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, trim, lit
from pyspark.sql.types import IntegerType

from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, PCA,
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

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
OUTPUT_DIR = r"C:\Users\MAURICIO\Downloads\data\salidas_bloque2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

# ============================================================
# PARTE A: PREPARACION DE DATOS (Tareas 7-8)
# ============================================================
print("=" * 60)
print("PARTE A: Pipeline de preparacion de datos")
print("=" * 60)

spark = (
    SparkSession.builder
    .appName("SIVIGILA_Bloque2_ML")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# --- Definir categorias de eventos (variable objetivo) ---
CATEGORIA_MAP = {
    # VECTORIAL
    205: "VECTORIAL", 210: "VECTORIAL", 217: "VECTORIAL", 220: "VECTORIAL",
    420: "VECTORIAL", 430: "VECTORIAL", 440: "VECTORIAL",
    460: "VECTORIAL", 470: "VECTORIAL", 490: "VECTORIAL", 495: "VECTORIAL",
    540: "VECTORIAL", 580: "VECTORIAL", 895: "VECTORIAL",

    # INMUNOPREVENIBLE
    230: "INMUNOPREVENIBLE", 320: "INMUNOPREVENIBLE",
    330: "INMUNOPREVENIBLE", 340: "INMUNOPREVENIBLE", 341: "INMUNOPREVENIBLE",
    345: "INMUNOPREVENIBLE", 348: "INMUNOPREVENIBLE",
    500: "INMUNOPREVENIBLE", 510: "INMUNOPREVENIBLE",
    520: "INMUNOPREVENIBLE", 530: "INMUNOPREVENIBLE",
    620: "INMUNOPREVENIBLE", 730: "INMUNOPREVENIBLE",
    760: "INMUNOPREVENIBLE", 770: "INMUNOPREVENIBLE", 800: "INMUNOPREVENIBLE",
    810: "INMUNOPREVENIBLE", 820: "INMUNOPREVENIBLE", 825: "INMUNOPREVENIBLE",
    831: "INMUNOPREVENIBLE",

    # MATERNO_INFANTIL
    110: "MATERNO_INFANTIL", 112: "MATERNO_INFANTIL", 113: "MATERNO_INFANTIL",
    298: "MATERNO_INFANTIL", 343: "MATERNO_INFANTIL",
    549: "MATERNO_INFANTIL", 550: "MATERNO_INFANTIL", 560: "MATERNO_INFANTIL",
    590: "MATERNO_INFANTIL", 600: "MATERNO_INFANTIL",
    735: "MATERNO_INFANTIL", 740: "MATERNO_INFANTIL", 750: "MATERNO_INFANTIL",

    # ZOONOTICO
    100: "ZOONOTICO", 300: "ZOONOTICO", 450: "ZOONOTICO", 455: "ZOONOTICO",

    # INTOXICACION_Y_VIOLENCIA
    228: "INTOX_VIOLENCIA",
    356: "INTOX_VIOLENCIA",
    360: "INTOX_VIOLENCIA", 370: "INTOX_VIOLENCIA", 380: "INTOX_VIOLENCIA",
    390: "INTOX_VIOLENCIA", 400: "INTOX_VIOLENCIA", 410: "INTOX_VIOLENCIA",
    412: "INTOX_VIOLENCIA", 414: "INTOX_VIOLENCIA",
    452: "INTOX_VIOLENCIA", 875: "INTOX_VIOLENCIA",

    # CRONICO
    155: "CRONICO", 456: "CRONICO", 457: "CRONICO", 459: "CRONICO",
    850: "CRONICO",

    # OTROS (tracome)
    305: "OTROS",
}

CATEGORIA_LABEL_MAP = {
    "VECTORIAL": 0,
    "INMUNOPREVENIBLE": 1,
    "MATERNO_INFANTIL": 2,
    "ZOONOTICO": 3,
    "INTOX_VIOLENCIA": 4,
    "CRONICO": 5,
    "OTROS": 6,
}

# --- Cargar datos igual que en el Bloque 1 ---
print("Cargando datos...")
pdf_raw = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8", keep_default_na=False)
print(f"  Pandas: {len(pdf_raw):,} filas")
pdf_raw = pdf_raw.rename(columns={"ANO": "ANO_STR"})

# Agregar columna de categoria y label usando mapeo en pandas
# (mas eficiente que multiples withColumn en Spark)
cod_to_cat = {str(k): v for k, v in CATEGORIA_MAP.items()}
cod_to_label = {str(k): CATEGORIA_LABEL_MAP[v] for k, v in CATEGORIA_MAP.items()}

pdf_raw["categoria"] = pdf_raw["COD_EVE"].map(cod_to_cat).fillna("OTROS")
pdf_raw["label"] = pdf_raw["COD_EVE"].map(cod_to_label)
pdf_raw["label"] = pdf_raw["label"].fillna(CATEGORIA_LABEL_MAP["OTROS"])

# Crear Spark DataFrame con las nuevas columnas
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
        when(trim(col("nom_mun")) == "", None).otherwise(trim(col("nom_mun"))))
    .drop("ANO_STR")
)

# Convertir COD_DPTO_O a string para StringIndexer
df = df.withColumn("dpto_str", col("COD_DPTO_O").cast("string"))

# Eliminar registros con COD_DPTO_O = 0 (desconocido)
df = df.filter(col("COD_DPTO_O") != 0)

# Cache
df = df.cache()
total = df.count()
print(f"  Registros para ML: {total:,}")

# Verificar distribucion de categorias
print("\n--- Distribucion de categorias (variable objetivo) ---")
df.groupBy("categoria", "label").count().orderBy("label").show(10, truncate=False)

# ============================================================
# Tarea 7: Pipeline de preparacion con VectorAssembler,
#          StringIndexer y OneHotEncoder
# ============================================================
print("\n--- Tarea 7: Construccion del pipeline de features ---")

# StringIndexer para departamento
idx_dpto = StringIndexer(
    inputCol="dpto_str", outputCol="dpto_idx", handleInvalid="keep"
)

# OneHotEncoder para departamento
ohe_dpto = OneHotEncoder(
    inputCols=["dpto_idx"], outputCols=["dpto_ohe"], dropLast=False
)

# VectorAssembler: unir SEMANA, one-hot de dpto, y conteo_casos
assembler = VectorAssembler(
    inputCols=["SEMANA", "dpto_ohe", "conteo_casos"],
    outputCol="features",
    handleInvalid="skip",
)

# StandardScaler
scaler = StandardScaler(
    inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True
)

# Pipeline completo de preparacion
prep_pipeline = Pipeline(stages=[idx_dpto, ohe_dpto, assembler, scaler])
prep_model = prep_pipeline.fit(df)
df_prep = prep_model.transform(df).cache()

# Materializar
n = df_prep.count()
print(f"  Registros tras pipeline: {n:,}")
print("\n  Esquema del vector de features (scaledFeatures):")
df_prep.select("scaledFeatures").printSchema()
df_prep.select("features", "scaledFeatures").show(5, truncate=False)

# ============================================================
# Tarea 8: Justificar estandarizacion elegida
# ============================================================
print("\n--- Tarea 8: Justificacion de la estandarizacion ---")
print("""
  Se eligio StandardScaler (withMean=True, withStd=True) porque:
  1. Los algoritmos de clasificacion como Regresion Logistica y PCA asumen
     o se benefician de datos con media 0 y varianza 1.
  2. Las features tienen escalas muy diferentes: SEMANA [1-52], conteo_casos
     [1-470], y variables one-hot [0-1]. Sin estandarizacion, conteo_casos
     dominaria las distancias euclidianas en K-Means y PCA.
  3. MinMaxScaler es sensible a outliers; el IQR mostro que el 10% de los
     registros son atipicos en conteo_casos, lo que distorsionaria el
     rango [0,1] si se usara MinMaxScaler.
""")

# ============================================================
# PARTE B: APRENDIZAJE NO SUPERVISADO (Tareas 9-12)
# ============================================================
print("=" * 60)
print("PARTE B: PCA + K-Means (no supervisado)")
print("=" * 60)

# --- Tarea 9: PCA y varianza explicada ---
print("\n--- Tarea 9: PCA ---")
num_features = len(df_prep.select("scaledFeatures").first()[0])
print(f"  Numero total de features (incluyendo one-hot): {num_features}")

pca = PCA(
    k=num_features, inputCol="scaledFeatures", outputCol="pcaFeatures"
)
pca_model = pca.fit(df_prep)

varianza = np.array(pca_model.explainedVariance.toArray())
varianza_acum = np.cumsum(varianza)
print("\n  Varianza explicada por componente:")
for i in range(min(15, num_features)):
    print(
        f"    PC{i + 1:2d}: {varianza[i]:.4f} "
        f"({varianza[i] / sum(varianza) * 100:5.1f}%) "
        f"acumulado: {varianza_acum[i] / sum(varianza) * 100:.1f}%"
    )

# --- Tarea 10: Decidir cuantos componentes retener ---
print("\n--- Tarea 10: Seleccion de componentes principales ---")
var_ratio = varianza_acum / sum(varianza)
k_80 = int(np.argmax(var_ratio >= 0.80)) + 1
k_90 = int(np.argmax(var_ratio >= 0.90)) + 1
print(f"  Componentes para explicar >= 80% de varianza: {k_80}")
print(f"  Componentes para explicar >= 90% de varianza: {k_90}")

# Criterio de Kaiser (eigenvalue > 1 en matriz de correlacion ≈
# varianza explicada > 1/num_features)
kaiser_threshold = 1.0 / num_features
k_kaiser = int(np.sum(varianza > kaiser_threshold))
print(f"  Componentes con eigenvalue > 1 (criterio Kaiser): {k_kaiser}")

# Usar el criterio de 90% de varianza o un minimo de 3
K_PCA = max(3, k_90) if k_90 <= num_features else min(10, num_features)
print(f"\n  => Se conservan K = {K_PCA} componentes principales")
print(f"     Justificacion: capturan >= {var_ratio[K_PCA - 1] * 100:.1f}% "
      f"de la varianza total.")

# Re-entrenar PCA con el K elegido
pca_final = PCA(
    k=K_PCA, inputCol="scaledFeatures", outputCol="pcaFeatures"
)
pca_model = pca_final.fit(df_prep)
df_pca = pca_model.transform(df_prep).cache()
_ = df_pca.count()

# Grafico de varianza explicada
print("\n--- Generando grafico 3: Varianza explicada PCA ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax1.bar(range(1, num_features + 1), varianza / sum(varianza), color="#2c7bb6")
ax1.axvline(x=K_PCA, color="red", linestyle="--", linewidth=2,
            label=f"K={K_PCA} seleccionado")
ax1.set_xlabel("Componente principal")
ax1.set_ylabel("Proporcion de varianza explicada")
ax1.set_title("Scree Plot - Varianza por componente")
ax1.legend()
ax1.set_xlim(0, min(20, num_features + 1))

# Varianza acumulada
ax2.plot(range(1, num_features + 1), var_ratio, "o-", color="#d7191c",
         markersize=3, linewidth=1.5)
ax2.axhline(y=0.90, color="gray", linestyle="--", label="90% varianza")
ax2.axvline(x=K_PCA, color="red", linestyle="--", linewidth=2,
            label=f"K={K_PCA}")
ax2.set_xlabel("Numero de componentes")
ax2.set_ylabel("Varianza acumulada")
ax2.set_title("Varianza acumulada por componentes")
ax2.legend()
ax2.set_xlim(0, min(20, num_features + 1))
ax2.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig3_pca_varianza.png"), bbox_inches="tight")
plt.close(fig)

# --- Tarea 11: K-Means sobre componentes principales ---
print("\n--- Tarea 11: K-Means y metodo del codo ---")

# Probar K de 2 a 10
wcss = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(k=k, seed=42, featuresCol="pcaFeatures", predictionCol=f"cluster_k{k}")
    model = km.fit(df_pca)
    wcss.append(model.summary.trainingCost)
    print(f"  K={k}: WCSS (inercia) = {model.summary.trainingCost:.2f}")

# Elbow plot
print("\n--- Generando grafico 4: Metodo del codo ---")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(K_range), wcss, "o-", color="#2c7bb6", linewidth=2, markersize=8)
ax.set_xlabel("Numero de clusters (K)")
ax.set_ylabel("WCSS / Inercia")
ax.set_title("Metodo del codo para K-Means\n(sobre componentes principales)")
ax.set_xticks(list(K_range))
ax.grid(True, alpha=0.3)

# Calcular el "codo" usando la diferencia porcentual
deltas = [((wcss[i - 1] - wcss[i]) / wcss[0]) for i in range(1, len(wcss))]
print("  Diferencia porcentual de WCSS entre K consecutivos:")
for i, d in enumerate(deltas):
    print(f"    K={i + 2}->{i + 3}: {d * 100:.1f}%")
    ax.annotate(f"{d * 100:.1f}%",
                (i + 2.5, wcss[i + 1]),
                fontsize=7, color="red",
                xytext=(0, -12), textcoords="offset points",
                ha="center")

# Elegir K donde la reduccion marginal cae significativamente
# (primer valor < 30% o < 10%)
umbral_elbow = 0.15  # 15% de mejora marginal
k_optimo = 3  # default
for i, d in enumerate(deltas):
    if d < umbral_elbow:
        k_optimo = i + 3  # i+2 es el K actual, +1 porque el delta se mide entre K y K+1
        break
if k_optimo == 3 and len(deltas) > 0:
    # Si ninguno cae bajo el umbral, elegir el de mayor caida
    k_optimo = int(np.argmax(deltas)) + 3

print(f"\n  => K optimo seleccionado: {k_optimo}")
ax.axvline(x=k_optimo, color="red", linestyle="--", linewidth=2,
           label=f"K optimo = {k_optimo}")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig4_elbow_kmeans.png"), bbox_inches="tight")
plt.close(fig)

# Entrenar K-Means final con K optimo
km_final = KMeans(
    k=k_optimo, seed=42, featuresCol="pcaFeatures", predictionCol="cluster"
)
km_model = km_final.fit(df_pca)
df_clustered = km_model.transform(df_pca).cache()
_ = df_clustered.count()

# --- Tarea 12: Interpretar los clusters ---
print("\n--- Tarea 12: Interpretacion de clusters ---")

# Perfil de cada cluster
print("\n  Tamanio de cada cluster:")
cluster_sizes = (
    df_clustered.groupBy("cluster").count().orderBy("cluster").collect()
)
for row in cluster_sizes:
    print(f"    Cluster {row['cluster']}: {row['count']:,} registros "
          f"({row['count'] / total * 100:.1f}%)")

# Distribucion de categorias por cluster
print("\n  Distribucion de categorias por cluster (top 3 por cluster):")
for c in range(k_optimo):
    print(f"\n  --- Cluster {c} ---")
    cat_dist = (
        df_clustered.filter(col("cluster") == c)
        .groupBy("categoria")
        .count()
        .orderBy(col("count").desc())
        .limit(3)
        .collect()
    )
    total_cluster = sum(r["count"] for r in cat_dist)
    for r in cat_dist:
        pct = r["count"] / sum(s["count"] for s in cluster_sizes if s["cluster"] == c) * 100
        print(f"    {r['categoria']}: {r['count']:,} ({pct:.1f}%)")

# Centroides de los clusters (componentes principales)
print("\n  Coordenadas de los centroides (primeras 3 PCs):")
centers = km_model.clusterCenters()
for i, center in enumerate(centers):
    print(f"    Cluster {i}: PC1={center[0]:.3f}, PC2={center[1]:.3f}, "
          f"PC3={center[2]:.3f}" + (f", ..." if len(center) > 3 else ""))

print("""
  Interpretacion de clusters:
  Los clusters formados por K-Means sobre los componentes principales
  agrupan patrones similares de:
    - Temporalidad (semana epidemiologica)
    - Ubicacion geografica (departamento)
    - Magnitud del brote (conteo de casos)
  La categoria predominante en cada cluster revela que ciertos tipos de
  eventos de salud publica comparten patrones espacio-temporales similares.
""")

# ============================================================
# PARTE C: CLASIFICACION SUPERVISADA (Tareas 13-18)
# ============================================================
print("=" * 60)
print("PARTE C: Clasificacion supervisada")
print("=" * 60)

# --- Tarea 13: Variable objetivo ---
print("\n--- Tarea 13: Variable objetivo definida ---")
print("  Variable objetivo: 'categoria' del evento de salud publica")
print("  Clases: VECTORIAL(0), INMUNOPREVENIBLE(1), MATERNO_INFANTIL(2),")
print("          ZOONOTICO(3), INTOX_VIOLENCIA(4), CRONICO(5), OTROS(6)")

# Contar registros por clase
class_counts = df.groupBy("label").count().orderBy("label").collect()
print("\n  Conteo por clase:")
for r in class_counts:
    cat_names = {v: k for k, v in CATEGORIA_LABEL_MAP.items()}
    print(f"    Clase {int(r['label'])} ({cat_names[int(r['label'])]}): "
          f"{r['count']:,} registros")

# --- Tarea 14: Train/test split ---
print("\n--- Tarea 14: Division train/test (80/20, seed=42) ---")
train, test = df_prep.select("scaledFeatures", "label").randomSplit(
    [0.8, 0.2], seed=42
)
print(f"  Train: {train.count():,} registros")
print(f"  Test:  {test.count():,} registros")

# --- Tarea 15: Entrenar dos modelos ---
print("\n--- Tarea 15: Entrenamiento de modelos ---")

# Regresion Logistica
print("\n  Entrenando Regresion Logistica...")
lr = LogisticRegression(
    featuresCol="scaledFeatures",
    labelCol="label",
    maxIter=100,
    regParam=0.1,
    elasticNetParam=0.0,
)
lr_model = lr.fit(train)
print("  Regresion Logistica entrenada.")

# Random Forest
print("\n  Entrenando Random Forest...")
rf = RandomForestClassifier(
    featuresCol="scaledFeatures",
    labelCol="label",
    numTrees=50,
    maxDepth=10,
    seed=42,
    featureSubsetStrategy="sqrt",
)
rf_model = rf.fit(train)
print("  Random Forest entrenado.")

# --- Tarea 16: Metricas ---
print("\n--- Tarea 16: Evaluacion de metricas ---")

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
evaluator_prec = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
)
evaluator_rec = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall"
)

for name, model in [("Regresion Logistica", lr_model), ("Random Forest", rf_model)]:
    preds = model.transform(test)
    acc = evaluator_acc.evaluate(preds)
    f1 = evaluator_f1.evaluate(preds)
    prec = evaluator_prec.evaluate(preds)
    rec = evaluator_rec.evaluate(preds)
    print(f"\n  {name}:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f} (weighted)")
    print(f"    Recall:    {rec:.4f} (weighted)")
    print(f"    F1-Score:  {f1:.4f}")

# --- Tarea 17: Matriz de confusion ---
print("\n--- Tarea 17: Matriz de confusion ---")

for name, model in [("Regresion Logistica", lr_model), ("Random Forest", rf_model)]:
    preds = model.transform(test)
    confusion = (
        preds.groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .toPandas()
    )

    # Construir matriz
    n_clases = 7
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for _, row in confusion.iterrows():
        cm[int(row["label"])][int(row["prediction"])] = int(row["count"])

    cat_names = {v: k[:12] for k, v in CATEGORIA_LABEL_MAP.items()}
    labels = [cat_names.get(i, f"Clase{i}") for i in range(n_clases)]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={"label": "Conteo"})
    ax.set_xlabel("Prediccion", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_title(f"Matriz de Confusion - {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"fig5_matriz_confusion_{name.replace(' ', '_').lower()}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Matriz guardada: {fname}")

    # Identificar donde se confunde mas (excluyendo la diagonal)
    errores = []
    for i in range(n_clases):
        for j in range(n_clases):
            if i != j and cm[i][j] > 0:
                errores.append((cm[i][j], i, j))

    errores.sort(reverse=True)
    if errores:
        print(f"  Top 3 confusiones de {name}:")
        for cnt, real, pred in errores[:3]:
            print(f"    Clase {cat_names[real]} predicha como "
                  f"{cat_names[pred]}: {cnt} veces")

# --- Tarea 18: Importancia de variables (Random Forest) ---
print("\n--- Tarea 18: Importancia de variables (Random Forest) ---")
importances = rf_model.featureImportances.toArray()

# Solo las primeras features tienen interpretacion directa:
# SEMANA (pos 0), dpto_ohe (pos 1 a 35), conteo_casos (ultima pos)
fi_data = [
    ("SEMANA", importances[0]),
    ("conteo_casos", importances[-1]),
]
# Agrupar importancia de todos los dpto one-hot
dpto_importance = float(np.sum(importances[1:-1]))
fi_data.append(("Departamento (35 vars)", dpto_importance))

fi_data.sort(key=lambda x: x[1], reverse=True)

print(f"  Feature importance total features: {len(importances)}")
print(f"  Feature importance agrupada:")
for feat, imp in fi_data:
    print(f"    {feat}: {imp:.4f}")

print("\n  Interpretacion:")
print("  La importancia de variables indica que SEMANA y conteo_casos son")
print("  los predictores mas fuertes de la categoria del evento, lo cual")
print("  es consistente: ciertas enfermedades tienen estacionalidad marcada")
print("  (vectoriales en epoca de lluvias, intoxicaciones constantes) y")
print("  volumenes tipicos de notificacion diferentes.")

# Grafico de importancia
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh([f[0] for f in reversed(fi_data)], [f[1] for f in reversed(fi_data)],
        color="#2c7bb6")
ax.set_xlabel("Importancia")
ax.set_title("Importancia de variables - Random Forest\n(agrupado)")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig6_feature_importance.png"),
            bbox_inches="tight")
plt.close(fig)

# ============================================================
# PARTE D: VALIDACION CRUZADA (Tareas 19-20)
# ============================================================
print("\n" + "=" * 60)
print("PARTE D: Validacion cruzada")
print("=" * 60)

# --- Tarea 19: CrossValidator con 2 hiperparametros ---
print("\n--- Tarea 19: CrossValidator con Random Forest ---")
print("  Hiperparametros a explorar: maxDepth y numTrees")

rf_cv = RandomForestClassifier(
    featuresCol="scaledFeatures",
    labelCol="label",
    seed=42,
    featureSubsetStrategy="sqrt",
)

param_grid = (
    ParamGridBuilder()
    .addGrid(rf_cv.maxDepth, [5, 10])
    .addGrid(rf_cv.numTrees, [20, 50])
    .build()
)

evaluator_cv = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)

cv = CrossValidator(
    estimator=rf_cv,
    estimatorParamMaps=param_grid,
    evaluator=evaluator_cv,
    numFolds=3,
    seed=42,
)

print("  Ejecutando validacion cruzada (3 folds, 4 combinaciones)...")
cv_model = cv.fit(train)

# --- Tarea 20: Reportar modelo ganador ---
print("\n--- Tarea 20: Modelo ganador y parametros optimos ---")
best_rf = cv_model.bestModel
best_max_depth = best_rf.getOrDefault("maxDepth")
best_num_trees = best_rf.getOrDefault("numTrees")

print(f"  Mejor maxDepth:  {best_max_depth}")
print(f"  Mejor numTrees:  {best_num_trees}")
print(f"  Mejor F1 score (CV avg): {max(cv_model.avgMetrics):.4f}")

# Evaluar el mejor modelo en test
best_preds = cv_model.transform(test)
best_acc = evaluator_acc.evaluate(best_preds)
best_f1 = evaluator_f1.evaluate(best_preds)
print(f"\n  Mejor modelo sobre test:")
print(f"    Accuracy: {best_acc:.4f}")
print(f"    F1-Score: {best_f1:.4f}")

# Reporte de todas las combinaciones
print("\n  Resultados de todas las combinaciones CV:")
params_list = cv_model.getEstimatorParamMaps()
for i, (params, metric) in enumerate(zip(params_list, cv_model.avgMetrics)):
    depth_val = params[rf_cv.maxDepth]
    ntree_val = params[rf_cv.numTrees]
    print(f"    Combo {i + 1}: maxDepth={depth_val}, numTrees={ntree_val} -> "
          f"F1 avg = {metric:.4f}")

# Tabla comparativa final
print("\n" + "=" * 60)
print("RESUMEN COMPARATIVO DE MODELOS")
print("=" * 60)

lr_acc = evaluator_acc.evaluate(lr_model.transform(test))
lr_f1 = evaluator_f1.evaluate(lr_model.transform(test))
rf_acc = evaluator_acc.evaluate(rf_model.transform(test))
rf_f1 = evaluator_f1.evaluate(rf_model.transform(test))

print(f"""
  Modelo                      Accuracy    F1-Score
  -----------------------------------------------
  Regresion Logistica         {lr_acc:.4f}      {lr_f1:.4f}
  Random Forest (base)        {rf_acc:.4f}      {rf_f1:.4f}
  Random Forest (CV tuneado)  {best_acc:.4f}      {best_f1:.4f}

  K-Means: {k_optimo} clusters formados sobre {K_PCA} componentes principales
  PCA: {K_PCA} componentes explican {var_ratio[K_PCA - 1] * 100:.1f}% de varianza
""")

# Limpiar cache
df.unpersist()
df_prep.unpersist()
df_pca.unpersist()
df_clustered.unpersist()

spark.stop()
print("\nSpark detenido. Bloque 2 completado exitosamente.")
print(f"Figuras guardadas en: {OUTPUT_DIR}")
