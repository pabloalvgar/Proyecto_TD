# Proyecto_TD
# **Proyecto de Aprendizaje Automático y NLP: Análisis de Recetas de Cocina**

## **Introducción**
En este proyecto, se aplican técnicas de **procesamiento de lenguaje natural (NLP)** y **aprendizaje automático** para resolver una tarea de **regresión** sobre un conjunto de datos de recetas de cocina. Los documentos textuales se representarán utilizando distintas técnicas de vectorización y se compararán modelos clásicos y avanzados de aprendizaje automático.

---

## **Descripción del Conjunto de Datos**
El dataset proporcionado (“**full_format_recipes.json**”) contiene **20,130 entradas** de recetas de la web [epicurious.com](https://www.epicurious.com). Cada receta incluye las siguientes variables:

- **Texto**:
  - `directions`: Instrucciones para preparar la receta.
  - `categories`: Categorías asignadas al plato.
  - `desc`: Descripción breve de la receta.
  - `title`: Título de la receta.
- **Numéricas**:
  - `rating`: Puntuación dada por usuarios (variable objetivo para regresión).
  - `fat`: Grasa (en gramos).
  - `protein`: Proteína (en gramos).
  - `calories`: Calorías (en gramos).
  - `sodium`: Sodio (en gramos).
- **Otros**:
  - `ingredients`: Lista de ingredientes.
  - `date`: Fecha de publicación.

---

## **Objetivo del Proyecto**
El objetivo principal es resolver un problema de **regresión**, prediciendo la variable `rating` a partir del texto disponible y de otras variables adicionales. Para ello, se llevará a cabo lo siguiente:

1. Procesado y homogeneización de textos.
2. Representación vectorial de los documentos mediante:
   - **TF-IDF**
   - **Word2Vec** (promedio de embeddings de palabras)
   - **Embeddings contextuales basados en Transformers** (e.g., BERT, RoBERTa).
3. Entrenamiento y evaluación de modelos de regresión:
   - **Redes neuronales** implementadas con PyTorch.
   - Otro modelo clásico usando Scikit-learn (K-NN, Random Forest, SVM, etc).
4. Comparación con el **fine-tuning de un modelo preentrenado** (Hugging Face Transformers) adaptado a la tarea de regresión.

## **Metodología**

### **1. Análisis Exploratorio de Datos (EDA)**
- Visualización de la relación entre `rating` y `categories`.
- Identificación de la distribución de los datos.

### **2. Preprocesamiento de Texto**
- Tokenización, lematización y eliminación de stopwords (NLTK, SpaCy).
- Preparación de texto sin procesar para embeddings contextuales (Transformers).

### **3. Representación Vectorial**
- **TF-IDF**: Representación numérica basada en la frecuencia de términos.
- **Word2Vec**: Promedio de embeddings de palabras preentrenados.
- **Embeddings Contextuales**: Uso de BERT, RoBERTa, etc., a través de Hugging Face.

### **4. Modelos de Regresión**
- **Redes Neuronales** (PyTorch): Entrenamiento de una red neuronal simple.
- **Modelo Clásico** (Scikit-learn): Comparación con K-NN, Random Forest, SVM, etc.

### **5. Fine-Tuning de Transformers**
- Fine-tuning de un modelo preentrenado (e.g., BERT) con una cabeza de regresión.

### **6. Evaluación y Comparación de Resultados**
- Métricas: **RMSE**, **MAE**, **R^2**.
- Validación cruzada para evaluar el rendimiento.
- Comparación gráfica de los resultados obtenidos.

---

## **Extensión del Proyecto**
Para la extensión, se explorarán las siguientes mejoras:
1. **Resúmenes automáticos**: Uso de un **summarizer** preentrenado (Hugging Face) para resumir las instrucciones (`directions`).
2. **Generación de recetas**: Implementación de técnicas de **prompting** en modelos del lenguaje como **LLaMA** o **Mixtral** para generar nuevas recetas.
3. **Análisis de Bigramas y POS Tagging**: Uso de técnicas de NLP adicionales (NLTK).
4. **Comparación de embeddings contextuales**: Evaluar distintos modelos (BERT, RoBERTa, DistilBERT).
5. **Visualización con Grafos**: Análisis de relaciones entre ingredientes o categorías usando técnicas basadas en grafos.

---

## **Tecnologías Utilizadas**
- **Lenguajes**: Python
- **Librerías**: Pandas, NumPy, Scikit-learn, PyTorch, Transformers (Hugging Face), Matplotlib, Seaborn, NLTK, SpaCy
- **Entorno**: Jupyter Notebook, Google Colab

---

## **Instalación**
1. Clona el repositorio:
   ```bash
   git clone https://github.com/usuario/proyecto_recetas_nlp.git
   cd proyecto_recetas_nlp
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta los notebooks o scripts en el entorno de tu elección.

---

## **Resultados Esperados**
- Comparación de modelos de regresi
