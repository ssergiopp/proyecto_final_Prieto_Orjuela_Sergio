FROM jupyter/pyspark-notebook:spark-3.5.0

USER root

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# Instalar librerias Python adicionales
RUN pip install --no-cache-dir \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    scikit-learn==1.4.2 \
    transformers==4.40.0 \
    torch==2.2.2 \
    pysentimiento==0.7.1 \
    huggingface_hub==0.22.2 \
    requests==2.31.0

# Crear directorios de trabajo
RUN mkdir -p /home/jovyan/work/data /home/jovyan/work/notebooks /home/jovyan/work/salidas

# Copiar archivos del proyecto
COPY data/ /home/jovyan/work/data/
COPY notebooks/ /home/jovyan/work/notebooks/
COPY salidas/ /home/jovyan/work/salidas/

# Asegurar permisos
RUN chown -R jovyan:users /home/jovyan/work

USER jovyan
WORKDIR /home/jovyan/work

EXPOSE 8888

CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
