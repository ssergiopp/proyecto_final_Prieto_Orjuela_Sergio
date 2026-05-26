"""
============================================================
BLOQUE 1 - ANALISIS EXPLORATORIO CON PySpark
Dataset: SIVIGILA - Vigilancia en Salud Publica, Colombia 2019
Parcial Final - Machine Learning con PySpark y Docker
============================================================
Ejecutar desde terminal:
    python bloque1_eda.py
"""
import os
import sys

# ============================================================
# Fix para PySpark en Windows + Java 17+:
# Evitar el error "getSubject is not supported" del Hadoop ViewFileSystem
# ============================================================
HADOOP_HOME = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "hadoop_tmp")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.makedirs(os.path.join(HADOOP_HOME, "bin"), exist_ok=True)
winutils_path = os.path.join(HADOOP_HOME, "bin", "winutils.exe")
if not os.path.exists(winutils_path):
    with open(winutils_path, "wb") as f:
        f.write(b"")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum as _sum, count, percentile_approx, when,
    regexp_replace, trim, round as spark_round, desc, asc,
    lit, collect_list
)
from pyspark.sql.types import IntegerType
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ============================================================
# CONFIGURACION
# ============================================================
CSV_PATH = r"C:\Users\MAURICIO\Downloads\data\sivigila.csv"
OUTPUT_DIR = r"C:\Users\MAURICIO\Downloads\data\salidas_bloque1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

# ============================================================
# TAREA 1: CARGAR DATASET CON PySpark
# ============================================================
print("=" * 60)
print("TAREA 1: Carga del dataset con PySpark")
print("=" * 60)

spark = (
    SparkSession.builder
    .appName("SIVIGILA_Bloque1_EDA")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Cargar con pandas primero (bypass Hadoop en Windows), luego a Spark
print("Cargando datos...")
pdf_raw = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8", keep_default_na=False)
print(f"  Pandas: {len(pdf_raw):,} filas cargadas")

pdf_raw = pdf_raw.rename(columns={"ANO": "ANO_STR"})
df = spark.createDataFrame(pdf_raw)

# Limpiar tipos en UNA sola cadena de transformaciones
df = (
    df
    .withColumn("conteo_casos", col("conteo_casos").cast(IntegerType()))
    .withColumn("SEMANA",       col("SEMANA").cast(IntegerType()))
    .withColumn("COD_DPTO_O",   col("COD_DPTO_O").cast(IntegerType()))
    .withColumn("COD_MUN_O",    col("COD_MUN_O").cast(IntegerType()))
    .withColumn("COD_EVE",      col("COD_EVE").cast(IntegerType()))
    .withColumn("ANO",
        regexp_replace(col("ANO_STR"), r"\.", "").cast(IntegerType()))
    .withColumn("nom_mun",
        when(trim(col("nom_mun")) == "", None).otherwise(trim(col("nom_mun"))))
    .drop("ANO_STR")
)

# CACHEAR para evitar recomputacion en cada accion
df = df.cache()

# Materializar cache con un conteo
total_registros = df.count()
print(f"  Registros totales: {total_registros:,}")
print(f"  Columnas: {len(df.columns)}")

print("\n--- Esquema ---")
df.printSchema()

print("\n--- Primeras 10 filas ---")
df.show(10, truncate=False)

# ============================================================
# TAREA 2: TRANSFORMACIONES ANALITICAS
# ============================================================
print("\n" + "=" * 60)
print("TAREA 2: Tres transformaciones analiticas")
print("=" * 60)

# T2a: filter - Top departamentos con mas DENGUE
print("\n--- T2a: Filter -- Top 10 departamentos con mas casos de DENGUE ---")
df_dengue = df.filter(col("Nombre") == "DENGUE").cache()
df_dengue_dpto = (
    df_dengue
    .groupBy("COD_DPTO_O")
    .agg(_sum("conteo_casos").alias("total_casos_dengue"))
    .orderBy(desc("total_casos_dengue"))
)
df_dengue_dpto.show(10)

# T2b: groupBy - Total casos por evento
print("\n--- T2b: GroupBy -- Total de casos por evento (top 15) ---")
df_evento_total = (
    df.groupBy("Nombre")
    .agg(_sum("conteo_casos").alias("total_casos"))
    .orderBy(desc("total_casos"))
)
df_evento_total.show(15, truncate=False)

# T2c: withColumn - Tasa de dengue por departamento (x1000)
print("\n--- T2c: WithColumn -- Tasa de dengue por 1000 casos del dpto ---")
total_por_dpto = df.groupBy("COD_DPTO_O").agg(_sum("conteo_casos").alias("total_dpto"))
df_tasa = (
    df_dengue_dpto
    .join(total_por_dpto, "COD_DPTO_O")
    .withColumn("tasa_dengue_x1000",
        spark_round(col("total_casos_dengue") / col("total_dpto") * 1000, 2))
    .orderBy(desc("tasa_dengue_x1000"))
)
df_tasa.show(10)

# ============================================================
# TAREA 3: ESTADISTICA DESCRIPTIVA
# ============================================================
print("\n" + "=" * 60)
print("TAREA 3: Estadistica descriptiva")
print("=" * 60)

print("\n--- describe() ---")
df.describe(["SEMANA", "COD_DPTO_O", "conteo_casos"]).show()

print("\n--- summary() conteo_casos ---")
df.select("conteo_casos").summary(
    "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
).show()

print("\n--- Percentiles 90, 95, 99 ---")
df.select(
    percentile_approx("conteo_casos", 0.90).alias("p90"),
    percentile_approx("conteo_casos", 0.95).alias("p95"),
    percentile_approx("conteo_casos", 0.99).alias("p99"),
).show()

print("\n--- Top 10 eventos mas frecuentes ---")
df.groupBy("Nombre").count().orderBy(desc("count")).show(10, truncate=False)

print("\n--- Casos totales por semana (primeras y ultimas 5) ---")
df_semanal = (
    df.groupBy("SEMANA")
    .agg(_sum("conteo_casos").alias("casos_semana"))
    .orderBy("SEMANA")
    .cache()
)
df_semanal.show(52)

# ============================================================
# TAREA 4: VALORES FALTANTES, DUPLICADOS Y ATIPICOS
# ============================================================
print("\n" + "=" * 60)
print("TAREA 4: Deteccion de datos faltantes, duplicados y atipicos")
print("=" * 60)

# Nulos en una sola pasada
print("\n--- Valores nulos por columna ---")
nulos_exprs = [
    _sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
]
nulos_row = df.agg(*nulos_exprs).collect()[0]
for c in df.columns:
    n = nulos_row[c]
    print(f"  {c}: {n} nulos ({(n / total_registros) * 100:.2f}%)")

# Duplicados
print("\n--- Deteccion de duplicados ---")
count_sin_dup = df.dropDuplicates().count()
duplicados = total_registros - count_sin_dup
print(f"  Duplicados exactos: {duplicados} ({(duplicados / total_registros) * 100:.2f}%)")

# Duplicados parciales (misma llave logica)
dup_llave = (
    df.groupBy("COD_EVE", "SEMANA", "COD_MUN_O")
    .agg(count("*").alias("n"))
    .filter(col("n") > 1)
)
dup_llave_count = dup_llave.count()
print(f"  Llaves duplicadas (COD_EVE+SEMANA+COD_MUN_O): {dup_llave_count} grupos")

# Atipicos
print("\n--- Valores atipicos en conteo_casos (IQR) ---")
stats = df.select(
    percentile_approx("conteo_casos", 0.25).alias("q1"),
    percentile_approx("conteo_casos", 0.75).alias("q3"),
).collect()[0]
q1, q3 = stats["q1"], stats["q3"]
iqr = q3 - q1
lim_sup = q3 + 1.5 * iqr
print(f"  Q1={q1}, Q3={q3}, IQR={iqr}, Limite superior={lim_sup}")

atipicos = df.filter(col("conteo_casos") > lim_sup).count()
print(f"  Registros atipicos (metodo IQR): {atipicos} ({(atipicos / total_registros) * 100:.2f}%)")

print("\n  Top 10 valores mas altos de conteo_casos:")
df.select("Nombre", "nom_mun", "SEMANA", "conteo_casos") \
  .orderBy(desc("conteo_casos")).show(10, truncate=False)

# COD_DPTO_O desconocido
print("\n--- Datos anomalos: COD_DPTO_O = 0 ---")
dpto_0_count = df.filter(col("COD_DPTO_O") == 0).count()
print(f"  Total registros con departamento desconocido (cod 0): {dpto_0_count}")
df.filter(col("COD_DPTO_O") == 0).show(5, truncate=False)

# ============================================================
# TAREA 5: VISUALIZACIONES (via Pandas)
# ============================================================
print("\n" + "=" * 60)
print("TAREA 5: Construccion de visualizaciones")
print("=" * 60)

print("--- Generando grafico 1: Top 15 eventos por total de casos ---")
pdf_eventos = df_evento_total.limit(15).toPandas().sort_values("total_casos")

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(pdf_eventos["Nombre"], pdf_eventos["total_casos"], color="#1f77b4")
ax.set_xlabel("Total de casos reportados", fontsize=11)
ax.set_title(
    "Top 15 eventos de salud publica por numero de casos\nSIVIGILA Colombia - 2019",
    fontsize=13, fontweight="bold",
)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, val in zip(bars, pdf_eventos["total_casos"]):
    ax.text(
        bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
        f"{val:,}", va="center", fontsize=8,
    )
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig1_top15_eventos.png"), bbox_inches="tight")
plt.close(fig)

print("--- Generando grafico 2: Serie temporal de DENGUE por semana ---")
pdf_dengue_semanal = (
    df_dengue
    .groupBy("SEMANA")
    .agg(_sum("conteo_casos").alias("casos_dengue"))
    .orderBy("SEMANA")
    .toPandas()
)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(
    pdf_dengue_semanal["SEMANA"], pdf_dengue_semanal["casos_dengue"],
    marker="o", linewidth=1.5, markersize=3, color="#d62728",
)
ax.fill_between(
    pdf_dengue_semanal["SEMANA"], pdf_dengue_semanal["casos_dengue"],
    alpha=0.15, color="#d62728",
)
ax.set_xlabel("Semana epidemiologica", fontsize=11)
ax.set_ylabel("Casos de dengue", fontsize=11)
ax.set_title(
    "Casos de dengue por semana epidemiologica\nColombia 2019",
    fontsize=13, fontweight="bold",
)
ax.set_xticks(range(1, 53, 4))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
z = pd.Series(pdf_dengue_semanal["casos_dengue"].values).rolling(4, center=True).mean()
ax.plot(
    pdf_dengue_semanal["SEMANA"], z,
    color="black", linewidth=2, linestyle="--",
    label="Tendencia (media movil 4 sem)",
)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig2_dengue_semanal.png"), bbox_inches="tight")
plt.close(fig)

print(f"  Figuras guardadas en: {OUTPUT_DIR}")

# ============================================================
# TAREA 6: CONCLUSIONES CUANTITATIVAS
# ============================================================
print("\n" + "=" * 60)
print("TAREA 6: Conclusiones cuantitativas")
print("=" * 60)

top3 = df_evento_total.limit(3).collect()
total_casos_global = df.agg(_sum("conteo_casos")).collect()[0][0]
top3_casos = sum(r["total_casos"] for r in top3)

dengue_pico = (
    df_dengue
    .groupBy("SEMANA")
    .agg(_sum("conteo_casos").alias("casos_dengue"))
    .orderBy(desc("casos_dengue"))
    .first()
)

dpto_top = (
    df.groupBy("COD_DPTO_O")
    .agg(_sum("conteo_casos").alias("total"))
    .orderBy(desc("total"))
    .first()
)

top10_dptos_sum = (
    df.groupBy("COD_DPTO_O")
    .agg(_sum("conteo_casos").alias("total"))
    .orderBy(desc("total"))
    .limit(10)
    .agg(_sum("total"))
    .collect()[0][0]
)

# Limpiar cache
df.unpersist()
df_dengue.unpersist()
df_semanal.unpersist()

print(f"""
CONCLUSION 1: Concentracion de morbilidad
  Los tres eventos de salud publica mas reportados en Colombia durante 2019 fueron:
    1. {top3[0]['Nombre']}: {top3[0]['total_casos']:,} casos
    2. {top3[1]['Nombre']}: {top3[1]['total_casos']:,} casos
    3. {top3[2]['Nombre']}: {top3[2]['total_casos']:,} casos
  En conjunto representan el {(top3_casos/total_casos_global)*100:.1f}% del total de
  {total_casos_global:,} casos reportados al sistema SIVIGILA. Esto indica que unas pocas
  patologias concentran la mayoria de la carga de notificacion obligatoria.

CONCLUSION 2: Estacionalidad del dengue
  El dengue muestra un patron estacional marcado en 2019. Los casos comienzan bajos en el
  primer trimestre, se incrementan progresivamente y alcanzan su pico maximo en la semana
  {dengue_pico['SEMANA']} ({dengue_pico['casos_dengue']:,} casos), coincidiendo con los periodos de mayor
  precipitacion en gran parte del territorio colombiano. A partir del cuarto trimestre los
  casos descienden, cerrando el ano con niveles bajos similares a los del inicio.

CONCLUSION 3: Concentracion geografica
  El departamento con codigo {dpto_top['COD_DPTO_O']} concentra la mayor cantidad de casos reportados
  ({dpto_top['total']:,} registros, {(dpto_top['total']/total_casos_global)*100:.1f}% del total nacional).
  Los 10 departamentos con mayor notificacion acumulan {(top10_dptos_sum/total_casos_global)*100:.1f}%
  de los casos, lo que sugiere que la capacidad de vigilancia epidemiologica y la carga de
  enfermedad estan fuertemente concentradas en centros urbanos y departamentos con mayor
  infraestructura de salud.

================================================================
RESUMEN DE ENTREGABLES
  - Script: bloque1_eda.py
  - Figuras: {OUTPUT_DIR}
      fig1_top15_eventos.png
      fig2_dengue_semanal.png
  - Dataset analizado: {CSV_PATH}
  - Total de registros procesados: {total_registros:,}
================================================================
""")

spark.stop()
print("Spark detenido. Bloque 1 completado exitosamente.")
