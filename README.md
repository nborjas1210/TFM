<h1 align="center"> 🤖 Lucy - Asistente Virtual para Vendedores de Productos Electronicos.  </h1>

 <p align="center">
  <img src="https://github.com/user-attachments/assets/6593f632-6fdb-4839-9d9a-c01a5f5349b8" alt="chatbot_lucy">
</p>

Este trabajo hace parte del Trabajo Final de Master de Adriana Sanchez Tinoco, Anderson Adrian Viscarra Alarcon y Neil Enrique Borjas Ramos, Universidad Internacional de la Rioja.
Master Universitario en Analisis y Visualizacion de Datos Masivos. 


**Lucy** es una asistente virtual inteligente diseñada para ayudar a vendedores de productos electrónicos, proporcionando información clave y visualizaciones relevantes sobre los productos de su interés.  

Esta herramienta combina:  

✔️ **Procesamiento de Lenguaje Natural (NLP - Natural Language Processing)**  
✔️ **Modelos de Lenguaje de Gran Escala (LLM - Large Language Models)**  
✔️ **Inteligencia Artificial Conversacional**  
✔️ **Machine Learning**  
✔️ **Ciencia y Análisis de Datos**  

---

## 🚀 Características Principales  

✔️ **Recomendación de Productos** - Basado en popularidad, rating y reseñas.  
✔️ **Análisis de Sentimiento** - Evaluación de opiniones de clientes sobre productos. La escala está vinculada con el sentimiento (positivo, neutro o negativo)*.  
✔️ **Comportamiento del Rating** - Análisis de datos históricos para identificar tendencias futuras de interés para el vendedor.  
✔️ **Reconocimiento de Nombres** - Personalización de la interacción con el usuario.  
✔️ **Análisis de Palabras Claves** - Generación de nubes de palabras con términos más usados en reseñas por producto.  
✔️ **Respuestas Conversacionales Naturales** - Integración con **DialoGPT**.  
✔️ **Gráficos y Visualizaciones** - Uso de **Matplotlib**, **Seaborn** y **Plotly**.  
✔️ **Sistema de Reglas y Respuestas** - Personalización de respuestas.  
✔️ **Idioma** - Se decidió crear el asistente virtual en inglés por alcance global y base de datos seleccionada.  

📖 ***Nota**:  Ramachandran, R., Sudhir, S., & Unnithan, A. B. (2021). *Exploring the relationship between emotionality and product star ratings in online reviews.*  
🔗 [Artículo en ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0970389621001178)  

---

## 🛠️ Tecnologías Utilizadas  

- **🤖 Rasa Framework** - Para procesamiento de lenguaje natural (NLU) y gestión de diálogos.  
- **📊 Pandas & Matplotlib** - Para manipulación y visualización de datos.  
- **🧠 DialoGPT** - Para generar respuestas conversacionales naturales.  
- **🔍 RapidFuzz** - Para mejorar la precisión en la búsqueda de productos.  
- **🎨 Plotly & Seaborn** - Para gráficos interactivos y análisis visual.  
- **☁️ WordCloud** - Para generación de nubes de palabras.  
- **⚡ Transformers** - Uso de modelos pre-entrenados de procesamiento de lenguaje natural.  

---
## 🏗️ **Estructura del Proyecto**  

📁 **/data** - Contiene los archivos YAML de NLU, rules y stories.  
📁 **/models** - Almacena los modelos entrenados de Rasa.  
📁 **/actions** - Código para acciones personalizadas en Rasa.  
📁 **/config.yml** - Configuración del pipeline de procesamiento.  
📁 **/domain.yml** - Definición de intenciones, acciones, entidades, slots y respuestas del bot.  
📁 **/nlu.yml** - Datos de entrenamiento para el procesamiento de lenguaje natural. Comprende intenciones y ejemplos de estas.  
📁 **/stories.yml** - Historias para la interacción del bot junto con las intenciones y acciones relacionadas.  
📁 **/rules.yml** - Reglas de comportamiento del bot.  
📁 **/endpoints.yml** - Configuración de servidores y API.  
📁 **/credentials.yml** - Credenciales de conexión a plataformas de mensajería.  

---

## 🔧 **Instalación y Uso**  

### 1️⃣ Paso
Instalar Anaconda desde su pagina oficial

### 2️⃣ Paso
Una vez instalado, selecciona Anaconda_prompt y ejecuta los siguientes comandos: 
```bash
conda create --name rasa-chatbot python=3.9
conda activate rasa-chatbot  #para activar el entorno.
```

### 3️⃣ Paso
Estando en el entorno, instala rasa y spacy con los siguientes comandos. Spacy brindara soporte adicional para el procesamiento de lenguaje natural: 
```bash
pip install rasa
pip install spacy
```

### 4️⃣ Paso. 
En este caso se le puede llamar Lucy_chatbot pero para un usuario principiante, se recomienda rasa-demo
```bash
cd Desktop
Mkdir rasa-demo
cd rasa-demo
Rasa init
```
Con rasa init se crea la estructura básica de un proyecto en RASA. Incluye ejemplos de archivos YAML como domain.yml, nlu.yml


### 5️⃣ Paso
Lo anterior hace también que el sistema ofrezca la opcion de entrenar el modelo inicial y probarlo. Es así como, cuando pregunte: 
Do you want to train the initial model, seleccionar yes.  
Cuando termine, el modelo habrá sido guardado. 


### 6️⃣ Paso 
Cuando pregunte: Do you want to speak to the trained Assistant on the command line? Seleccionar Yes. 


### 7️⃣ Paso
Para salir del asistente, se hace con /stop. 

**Nota**
De ahi en adelante, cuando se requiera ingresar nuevamente, se deben de abrir dos ventanas con el anaconda prompt. 
En una, debes quedar ubicado en la carpeta donde se encuentra tu proyecto, y correr los siguientes comandos.
```bash
conda activate rasa-chatbot
rasa train
rasa shell
```
En otra, debes quedar ubicado en la carpeta donde se encuentra tu proyecto, y correr los siguientes comandos para activar el servidor de acciones.
```bash
conda activate rasa-chatbot
rasa run actions
```
---

## 🏆 **Casos de Uso**  

### 📌 Ejemplo 1: Producto con mejores calificaciones  

🗨 **Usuario:**  
*"What is the product with the highest reviews and best rates?"*  
_(¿Cuál es el producto más popular con más comentarios y mejores calificaciones?)_  

🤖 **Lucy:**  
*"The product with the highest number of reviews and best rating is:"*  

```
Product Name: Apple AirTag
Parent ASIN: B0051VVOB2
Average Rating: 3.8
Number of reviews: 50
```

---

### 📌 Ejemplo 2: Análisis de Sentimiento  

🗨 **Usuario:**  
*"Show the sentiment for Apple AirTag."*  

🤖 **Lucy:**  
*"The sentiment gauge for Apple AirTag has been successfully generated and saved. Based on the following range:"*  

```
🔹 **Rating Range | Sentiment Classification**
❌ 1.0 - 1.9  → Negative
⚠️ 2.0 - 2.9  → Negative with hints of neutrality
🔘 3.0 - 3.9  → Mixed or neutral
✅ 4.0 - 4.4  → Positive with slight moderation
💚 4.5 - 5.0  → Positive
```


---

## 📌 **Intenciones Soportadas - Intents**  

Lucy puede entender y responder a diversas consultas como:  

✔️ Saludos y despedidas (`greet`, `bye`)  
✔️ Estado de ánimo (`mood_great`, `mood_unhappy`)  
✔️ Preguntas sobre su creador (`who_creator`)  
✔️ Solicitud de productos más vendidos (`get_topproduct`)  
✔️ Recomendaciones por popularidad, ofreciendo 10 productos basados en comentarios y rating (`recommend_bypopularity`)  
✔️ Distribucion de rating (`ask_product_ratings`)  
✔️ Comportamiento de ratings a lo largo del tiempo (`ask_product_rating_behavior`)  
✔️ Generación de gráficos de sentimiento (`request_sentimentgauge`)  
✔️ Extracción de palabras clave en reseñas y formacion de una nube de palabras (`ask_common_words`)  
✔️ Respuestas conversacionales con **DialoGPT** (`ask_dialoqpt_response`)  

---

## 📜 **Referencia**  
**Base de datos:**
Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952. Recuperado de https://amazon-reviews-2023.github.io/ 
**Analisis de Sentimiento:**
Ramachandran, R., Sudhir, S., & Unnithan, A. B. (2021). *Exploring the relationship between emotionality and product star ratings in online reviews.*  
🔗 [Artículo en ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0970389621001178)  

---
