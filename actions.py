# This files contains your custom actions which can be used to run
# custom Python code.


# 1. Importing Libraries - importar librerias
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import re
import time
import random
import numpy as np
import torch
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from rapidfuzz import fuzz, process 
from typing import Any, Text, Dict, List
from wordcloud import WordCloud
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# 1. This is to respond to questions based on DialoGPT and to keep a record of conversations for better conversations. 
# 1. Esta accion fue creada para responder preguntas basadas en DialoGPT y para mantener un record de las conversaciones, para interacciones coherentes
class ActionDialoGPTResponse(Action):
    def name(self) -> str:
        return "action_dialoqpt_response"

    def __init__(self):
        # Cargar modelo y el tokenizer de DialoGPT (Para asegurar la compatibilidad con el modelo de DialoGPT-Large, usando un tokenizer pre-entrenado. )
        # Load the model and DialoGPT Tokenizer (To ensure the compatibility with the DialoGPT-Large Model, using a pre-trained tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.chat_history_ids = None

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        user_input = tracker.latest_message.get("text")

        # Tokenizar la entrada del usuario - Tokenize the user's input. 
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')

        # Mantener el historial de la conversación - This is to keep the records of any conversation. 
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Generar la respuesta usando el modelo - Generate the response using the model
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.75
        )

        # Decodificar la respuesta generada - Decoding the generated response
        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Enviar la respuesta al usuario - Send the response to the user. 
        dispatcher.utter_message(text=response)
        return []


#2.  Name Recognition - Reconocimiento de Nombre.
class SetPersonName(Action):
    def name(self) -> str:
        return "action_set_person_name"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        name = None
        entities = tracker.latest_message.get("entities", [])
        for entity in entities:
            if entity["entity"] == "PERSON":
                name = self.clean_name(entity["value"])
                break

        if name:
            dispatcher.utter_message(text=f"That's a very nice name, {name}. Nice to meet you!")
            return [SlotSet("PERSON", name)]
        else:
            dispatcher.utter_message(text="I couldn't catch your name. Could you please repeat?")
            return []

    @staticmethod
    def clean_name(raw_name: str) -> str:
        match = re.search(r"\b[A-Za-zÀ-ÖØ-öø-ÿ'-]+\b", raw_name)
        return match.group(0) if match else raw_name
        
#3. fallback action activated when the assistant does not understand what the user is saying - Accion activa para cuando el asistente no entienda el mensaje del usuario. 
class ActionDefaultFallback(Action):
    def name(self):
       return "action_default_fallback"

    def run(self, dispatcher, tracker, domain):
       responses = [
           "I'm not sure how to respond to that, but I'd love to help you with something else!",
           "That's a tricky one! Can you rephrase or ask about something specific?",
           "Interesting! I'm learning every day. Is there something else you'd like to ask?",
           "Hmm, I didn't get that. Let's talk about something else!"
       ]
       dispatcher.utter_message(text=random.choice(responses))
       return []

#4. Top Product- The number 1 based on rating and reviews. // Producto Top #1 Basado en rating y comentarios. 
class TopProductsAction(Action):
    def name(self) -> Text:
        return "action_get_topproducts"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
       
        csv_path = r"C:\Users\ASANCHEZTI\OneDrive\Desktop\final_cleaned_sample_vf.csv"

        try:
            
            csv_data = pd.read_csv(csv_path)

            # Ensuring timestamp is in the correct format - Aseguramos que el timestamp este en el formato correcto.
            if "timestamp" in csv_data.columns:
                csv_data["timestamp"] = pd.to_datetime(csv_data["timestamp"], errors="coerce")

            # Group by 'parent_asin' to calculate metrics - Se agrupan los parent_asin para el calculo de las metricas.
            grouped_data = (
                csv_data.groupby("parent_asin")
                .agg(
                    avg_rating=("rating", "mean"),  # average rating - Rating Promedio
                    review_count=("rating", "count"),  # Number of reviews - Numero de Comentarios
                    product_title=("product_title", "first")  #Using the first product_title -  Usando el primer product_title
                )
                .reset_index()
            )

            # Filtering data for products with avg_rating higher or equal to 4 - Se filtran los datos para los productos que tengan un promedio mayor o igual a 4
            filtered_data = grouped_data[grouped_data["avg_rating"] >= 4]

            # Find the product with the highest number of reviews - Se encuentra el producto con el mayor numero de comentarios. 
            if not filtered_data.empty:
                top_product = filtered_data.loc[filtered_data["review_count"].idxmax()]
                parent_asin = top_product["parent_asin"]
                avg_rating = round(top_product["avg_rating"], 2)
                review_count = top_product["review_count"]
                product_title = top_product["product_title"]

                # Prepare the message with the outcome of calculations, indicating the Top 1 product
                # Se prepara el mensaje con el resultado de los calculos, indicando el producto # 1. 
                message = (
                    f"The product with the highest number of reviews and best rating is:\n"
                    f"Product Name: {product_title}\n"
                    f"Parent ASIN: {parent_asin}\n"
                    f"Average Rating: {avg_rating}\n"
                    f"Number of Reviews: {review_count}"
                )
            else:
                # Handle the case when no products meet the condition - Se considera el caso de cuando el producto no cumple con la condicion. 
                message = "No products meet the criteria of an average rating >= 4 and having the highest number of reviews."

            dispatcher.utter_message(text=message)

        except FileNotFoundError:
            # Handle the case when the dataset is not found - Se maneja en el caso dado que el dataset no sea encontrado. 
            dispatcher.utter_message(
                text="The dataset file could not be found. Please ensure the file exists at the specified path."
            )
        except Exception as e:
            # Handle general errors - Manejo de errores generales que pueden ocurrir. 
            print(f"Error during analysis: {e}")
            dispatcher.utter_message(
                text="An error occurred while analyzing the dataset. Please try again."
            )

        return []

#5. Recommending Top 10 Products Based on Popularity - Recomendacion de Productos dentro del Top 10 basados en Popularidad.
class ActionRecommendationByPopularity(Action):
    def name(self) -> Text:
        return "action_recommend_by_popularity"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            #Load dataset - Carga del dataset.
            dataset_path = r"C:\Users\ASANCHEZTI\OneDrive\Desktop\final_cleaned_sample_vf.csv"
            df = pd.read_csv(dataset_path)

            # Ensure critical columns are present and clean - Se asegura que las columnas criticas esten presentes y limpias
            required_columns = ['parent_asin', 'rating', 'product_title']
            if not all(col in df.columns for col in required_columns):
                dispatcher.utter_message(text="The dataset is missing critical columns.")
                return []

            # Drop rows with missing critical data - Se elimina filas con valores nulos en columnas criticas. 
            df = df.dropna(subset=required_columns)

            # Convert 'rating' to numeric - Se convierte la columna 'rating; en numerico. 
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df = df.dropna(subset=['rating'])

            # Group by 'parent_asin' and calculate metrics - Se agrupa por "parent_asin" y se calculan las metricas. 
            grouped = (
                df.groupby('parent_asin')
                .agg(
                    avg_rating=('rating', 'mean'),
                    review_count=('rating', 'count'),
                    product_title=('product_title', 'first')
                )
                .reset_index()
            )

            # Filter for products with at least 10 reviews - Filtrando productos con al menos 10 comentarios.
            filtered = grouped[grouped['review_count'] >= 10]

            # Sort by popularity (number of reviews) and average rating - Se ordena por popularidad (Numero de comentarios) y el rating promedio
            top_products = filtered.sort_values(
                by=['review_count', 'avg_rating'], ascending=[False, False]
            )

            # Get top 10 of Products based on Popularity - Obtener el top 10 de Productos por Popularidad
            top_n = 10  # Change to 5 if you want only the top 5 // Cambiar a 5 si se requiere solo el top 5
            top_results = top_products.head(top_n)

            # Preparing response message in case it's empty - Preparando Mensaje de Respuesta en caso que este vacio
            if top_results.empty:
                dispatcher.utter_message(
                    text="No products meet the criteria for popularity and reviews."
                )
                return []
            #Prepararing response Message with the result. - Preparando el mensaje de respuesta con resultad
            response_message = "Here are the top products based on popularity and reviews:\n"
            for i, row in top_results.iterrows():
                response_message += (
                    f"{i + 1}. {row['product_title']} (ASIN: {row['parent_asin']})\n"
                    f"   - Average Rating: {row['avg_rating']:.2f}\n"
                    f"   - Number of Reviews: {row['review_count']}\n"
                )

            dispatcher.utter_message(text=response_message)

        except Exception as e:
            dispatcher.utter_message(
                text=f"An error occurred: {str(e)}"
            )

        return []


#6. Rating Distribution per Product Name - Distribucion del Rating de acuerdo al Nombre del ProductoE
class ActionProductRatings(Action):
    def name(self) -> str:
        return "action_product_ratings"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        try:
            dataset_path = os.getenv("DATASET_PATH", r"C:\Users\ASANCHEZTI\OneDrive\Desktop\final_cleaned_sample_vf.csv")
            if not os.path.exists(dataset_path):
                dispatcher.utter_message(text="Dataset file not found. Please check the file path.")
                return []
            df = pd.read_csv(dataset_path)

            # Extract user query - Extraccion de la consulta del usuario, el texto proporcionado.
            user_query = tracker.latest_message.get("text", "").strip()
            possible_products = tracker.get_slot("possible_products")

            # Handle numeric selection - Manejo de seleccion numerica.
            if possible_products and user_query.isdigit():
                selected_index = int(user_query) - 1
                if 0 <= selected_index < len(possible_products):
                    product_title = possible_products[selected_index]
                    filtered_data = df[df["product_title"] == product_title]

                    if not filtered_data.empty:
                        # If not filtered data is empty, Generate and send chart. Si la informacion filtrada no esta vacia, se procede a generar el grafico
                        return self.generate_chart(filtered_data, product_title, dispatcher)
                    else:
                        dispatcher.utter_message(text=f"No data found for '{product_title}'.")
                        return [SlotSet("possible_products", None)]

                dispatcher.utter_message(text="Invalid selection. Please try again.")
                return []

            # Fuzzy matching to find similar product titles - Se utiliza el Fuzzy Matching para encontrar nombres de productos similares.
            product_name = self.extract_product_name(user_query)
            if not product_name:
                dispatcher.utter_message(text="I couldn't understand the product name. Please rephrase.")
                return []

            product_titles = df["product_title"].dropna().tolist()
            matches = process.extract(product_name, product_titles, limit=10)

            # Filter matches with a confidence score above a threshold (70) - Se filtran las coincidencias con una confidencia que este por encima del umbral (70) 
            threshold = 70
            filtered_matches = []
            seen_asins = set()
            for match in matches:
                product_row = df[df["product_title"] == match[0]]
                if not product_row.empty:
                    asin = product_row["parent_asin"].iloc[0]
                    if asin not in seen_asins:
                        seen_asins.add(asin)
                        filtered_matches.append(match[0])
         
            #To show only the first 70 characters in case the name is too long - Se restringe a que muestre los primeros 70 caracteres cuando el nombre es muy largo
            #This was made so whenever the user asks for a product, the chatbot looks the closest options in the dataset and display it for more accurate selection. 
            #Esto fue hecho para cuando el usuario pregunte por un producto, el chatbot revise las opciones mas cercanas en el dataset y las muestre para una seleccion mas precisa. 
            if len(filtered_matches) > 1:
                options = "\n".join(
                    [f"{i + 1}. {title[:70] + '...' if len(title) > 70 else title}" for i, title in enumerate(filtered_matches)] 
                )
                dispatcher.utter_message(text=f"I found multiple matches. Please select one:\n{options}")
                return [SlotSet("possible_products", filtered_matches)]

            elif filtered_matches:
                product_title = filtered_matches[0]
                filtered_data = df[df["product_title"].str.strip().str.lower() == product_title.strip().lower()]

                if not filtered_data.empty:
                    # Generate and send chart
                    return self.generate_chart(filtered_data, product_title, dispatcher)      

            dispatcher.utter_message(text="No matching products found. Please try again.")
            return []

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred: {str(e)}")
            return []

    #Extracts product name from the user query - Extrae el nombre del producto desde la consulta hecha por el usuario 
    def extract_product_name(self, user_query: str) -> str:
        query = re.sub(
            r"^(please tell me about the ratings? of|show me the ratings? for|what are the ratings? for|ratings? for)\s*",
            "",
            user_query,
            flags=re.IGNORECASE,
        )
        return query.strip().lower()

    #Generating the bar chart with the rating distribution - Generacion del Grafico de Barras con la distibucion del rating. 
    def generate_chart(self, df: pd.DataFrame, product_title: str, dispatcher: CollectingDispatcher) -> list:
        try:
            # Ensure 'rating' is numeric
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            rating_counts = df["rating"].value_counts(normalize=False).reindex(range(1, 6), fill_value=0).sort_index()

            if rating_counts.sum() == 0:
                dispatcher.utter_message(text=f"No ratings available for '{product_title}'.")
                return [SlotSet("possible_products", None)]

            # Create the bar chart - Creando el grafico de Barras
            plt.figure(figsize=(8, 6))
            bars = plt.bar(rating_counts.index, rating_counts.values, color="skyblue", width=0.6)

            # Add data labels to each bar to inform the Qty of reviews under that category - Agregando las etiquetas en cada barra, para informar la cantidad de reviews sobre cierta categoria. 
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.5,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    color='black'
                )

            # Set chart titles and labels - Estableciendo los Titulos y Etiquetas 
            plt.title(f"Ratings Distribution for {product_title}")
            plt.xlabel("Rating")
            plt.ylabel("Number of Reviews")
            plt.xticks(range(1, 6))
            plt.ylim(0, max(5, rating_counts.max() + 1))

            # Save the chart - Se guarta dentro de una carpeta local llamada "Charts"
            folder_path = r"C:\Projects\lucy_chatbot\charts"
            os.makedirs(folder_path, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"{product_title.replace(' ', '_')}_ratings_chart_{timestamp}.png"
            file_path = os.path.join(folder_path, file_name)
            plt.savefig(file_path, format="png")
            plt.close()

            dispatcher.utter_message(text=f"The ratings distribution chart for '{product_title}' has been saved to: {file_path}")
            return [SlotSet("possible_products", None)]

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred while generating the chart: {str(e)}")
            return [SlotSet("possible_products", None)]
        
#7. Common Words on the reviews - Palabras Comunes en los reviews.  
class ActionCommonWords(Action):
    def name(self) -> str:
        return "action_common_words"
    
    #Executing the action. Dispatcher sends the responses to the user; Tracker saves the history of conversations and recover the information as values in the slots. 
    #Se ejecuta la accion. El Dispatcher permite la respuesta al usuario; El Tracker guarda el historial de conversacion y recupera la informacion como valores en los slots 
    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        # Get the product name from the tracker slot - Recupera el nombre del producto desde el slot del tracker.
        product_name = tracker.get_slot("product_name")
        dataset_path = os.getenv("DATASET_PATH", r"C:\Users\ASANCHEZTI\OneDrive\Desktop\final_cleaned_sample_vf.csv")

        if not product_name:
            dispatcher.utter_message(text="I couldn't identify the product. Please specify.")
            return []
        
        try:
            df = pd.read_csv(dataset_path)
            df["product_title"] = df["product_title"].str.lower()
            product_name = product_name.lower()

            # Filter for the product - Se filtra por producto
            product_reviews = df[df["product_title"].str.contains(product_name, na=False)]

            if product_reviews.empty:
                dispatcher.utter_message(text=f"No data found for {product_name}.")
                return []

            # Refined tokenization and extraction of irrelevant words - Tokenizacion refinada y extraccion de palabras irrelevantes
            def extract_relevant_words(text):
                # Extended list of irrelevant words
                irrelevant_words = {    
                    "a", "an", "the", "and", "or", "but", "if", "in", "on", "with", "at", "by", "for", "to", "of",
                    "this", "that", "it", "is", "was", "are", "were", "be", "been", "has", "have", "had", "not",
                    "no", "yes", "you", "i", "we", "they", "he", "she", "his", "her", "its", "our", "their", "my",
                    "mine", "yours", "ours", "theirs", "really", "absolutely", "already", "highly", "perfectly",
                    "quickly", "pretty", "mostly", "definitely", "initially", "finally", "surely", "lately",
                    "personally", "easily", "away", "may", "today", "currently", "normally", "extremely", 
                    "slightly", "minimally", "directly", "constantly", "very", "quite", "too", "enough", "somehow",
                    "seems", "probably", "likely", "sometimes", "often", "rarely", "always", "never", "almost", 
                    "just", "then", "though", "yet", "perhaps", "also", "however", "therefore", "instead", 
                    "maybe", "meanwhile", "thus", "besides", "otherwise", "furthermore", "indeed", "simply", 
                    "merely", "each", "either", "neither", "both", "who", "whom", "whose", "where", "when", 
                    "which", "these", "those", "any", "all", "many", "few", "most", "several", "such", "like", 
                    "etc", "etc.", "oh", "uh", "um", "hence", "now", "anyway", "since", "try", "basically", 
                    "dearly", "clunky", "monthly", "stay", "ability", "capability", "weekly", "pay"
                }
                words = re.findall(r'\b\w+\b', text.lower())
                relevant_words = [word for word in words if word not in irrelevant_words and word.endswith('y')]
                return relevant_words

            reviews = " ".join(product_reviews["text"].dropna().tolist())
            relevant_words = extract_relevant_words(reviews)

            # Generate word cloud - Generar la Nube de Palabras
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(relevant_words))

            # Save the word cloud in a local folder called feelingcloud- Se guarda la nube de palabras en una carpeta local llamada Feelingcloud
            folder_path = r"C:\Projects\lucy_chatbot\feelingcloud"
            os.makedirs(folder_path, exist_ok=True)
            file_name = f"{product_name.replace(' ', '_')}_feeling_cloud.png"
            file_path = os.path.join(folder_path, file_name)
            wordcloud.to_file(file_path)

            # Respond to the user - Respuesta al usuario. 
            dispatcher.utter_message(
                text=f"After analyzing the reviews, here are the most common words found:",
            )
            dispatcher.utter_message(image=file_path)

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred: {str(e)}")
        
        return []

#8  Sentiment Analysis. - Analisis de Sentimiento
"""
Reference/ Referencia: Exploring the relationship between emotionality and product star ratings in online reviews.
Authors/ Autores: Rahul Ramachandran, Subin Sudhir, Anandakuttan B. Unnithan.
This study examines how emotions in reviews impact product ratings, useful for sentiment analysis. Refer to Table A.3
Este estudio examina como las emociones en los comentarios impactan los ratings en los productos, util para el analisis de sentimiento. Hacer referencia a Tabla A.3
URL: https://www.sciencedirect.com/science/article/pii/S0970389621001178
"""
#Considering the document, a radiator was created indicating green as a possitive sentiment and red as a negative sentiment. 
#Considerando el documento, un radiador fue creado para indicar el verde como un sentimiento positivo fuerte y el rojo con un sentimiento negativo fuerte. 

# Classification of Averages in Relation to Sentiment / Clasificación de Promedios en Relación al Sentimiento:

#   ┌───────────────┬─────────────────────────────────────────────────────────────────────────────┐
#   │ Rating Range  │ Sentiment Classification / Clasificación del Sentimiento (Español)          │
#   ├───────────────┼───────────────────────────────────────────────────────────────── ───────────┤
#   │  1.0 - 1.9    │ ❌ Negative - Negativo                                                      │
#   │  2.0 - 2.9    │ ⚠️ Negative with hints of neutrality  - Negativo con atisbos de neutralidad │
#   │  3.0 - 3.9    │ 🔘 Mixed or neutral - Mixto o neutral                                       │
#   │  4.0 - 4.4    │ ✅ Positive with slight moderation - Positivo con ligera moderación         │
#   │  4.5 - 5.0    │ 💚 Positive Strong Feeling - Sentimiento Positivo Fuerte                    │
#   └───────────────┴─────────────────────────────────────────────────────────────────────────────┘


class ActionGenerateSentimentRadiator(Action):
    def name(self) -> str:
        return "action_generate_sentiment_gauge"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        try:
            dataset_path = os.getenv("DATASET_PATH", r"C:\Users\ASANCHEZTI\OneDrive\Desktop\final_cleaned_sample_vf.csv")
            
            if not os.path.exists(dataset_path):
                dispatcher.utter_message(text="Dataset not found. Please check the file path.")
                return []
            
            df = pd.read_csv(dataset_path)
            
            # Extract the product name from the user's input - Extraccion del nombre del producto desde el texto del usuario. 
            product_name = tracker.get_slot("product_name")
            if not product_name:
                dispatcher.utter_message(text="I couldn't identify the product. Please specify the product name.")
                return []
            
            # Normalize product names for matching - Normalizacion del nombre del producto para despues, encontrar coincidencias.
            df["product_title"] = df["product_title"].str.lower().str.strip()
            product_name = product_name.lower().strip()
            
            # Filter the dataset for the specified product - Filtrar el dataset para el producto en especifico. 
            product_data = df[df["product_title"].str.contains(product_name, na=False)]
            
            if product_data.empty:
                dispatcher.utter_message(text=f"No data found for the product '{product_name}'.")
                return []
            
            # Calculate the average rating for the product - Calculo del rating promedio.
            avg_rating = product_data["rating"].mean()
            
            # Define a feeling gauge chart - Defino el grafico de Indicador de sentimientos.
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_rating,
                title={'text': f"Sentiment for {product_name.title()}"},
                gauge={
                    'axis': {'range': [0, 5]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 2], 'color': "red"},
                        {'range': [2, 3], 'color': "orange"},
                        {'range': [3, 4], 'color': "yellow"},
                        {'range': [4, 5], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "blue", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_rating
                    }
                }
            ))

            # Save the feeling gauge chart - Guardo el grafico de Indicador de sentimientos
            folder_path = r"C:\Projects\lucy_chatbot\gaugefeeling"
            os.makedirs(folder_path, exist_ok=True)
            file_name = f"radiator_{product_name.replace(' ', '_')}.png"
            file_path = os.path.join(folder_path, file_name)
            fig.write_image(file_path)

            # Respond to the user - Respuesta al usuario. 
            dispatcher.utter_message(text=(f"The sentiment gauge for '{product_name}' has been successfully generated and saved. Based on the following range:\n\n"
                "Classification of Averages in Relation to Sentiment:\n\n"
                "🔹 **Rating Range | Sentiment Classification**\n"
                "❌ **1.0 - 1.9**  → Negative\n"
                "⚠️ **2.0 - 2.9**  → Negative with hints of neutrality\n"
                "🔘 **3.0 - 3.9**  → Mixed or neutral\n"
                "✅ **4.0 - 4.4**  → Positive with slight moderation\n"
                "💚 **4.5 - 5.0**  → Positive\n\n"
            ))
            dispatcher.utter_message(image=file_path)
        
        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred while generating the sentiment radiator: {str(e)}")
        
        return []

#9. Rating behavior over time - Comportamiento del rating en el tiempo  
class ActionProductRatingBehavior(Action):
    def name(self) -> str:
        return "action_product_rating_behavior"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        try:
            dataset_path = os.getenv("DATASET_PATH", r"C:\Users\ASANCHEZTI\OneDrive\Desktop\final_cleaned_sample_vf.csv")
            df = pd.read_csv(dataset_path)

            # Extract user query - Extraccion de la consulta del usuario
            user_query = tracker.latest_message.get("text", "").strip()
            possible_products = tracker.get_slot("possible_products")

            # Possible Options for the user to choose - Opciones posibles para que el usuario seleccione
            if possible_products and user_query.isdigit():
                selected_index = int(user_query) - 1
                if 0 <= selected_index < len(possible_products):
                    product_title = possible_products[selected_index]
                    return self.generate_chart(df, product_title, dispatcher)
                else:
                    dispatcher.utter_message(text="Invalid selection. Please try again.")
                    return []

            # Fuzzy matching to find similar product - Fuzzy Matching para encontrar productos similares
            product_name = self.extract_product_name(user_query)
            if not product_name:
                dispatcher.utter_message(text="I couldn't understand the product name. Please rephrase.")
                return []

            product_titles = df["product_title"].dropna().tolist()
            matches = process.extract(product_name, product_titles, limit=10)
            
            threshold = 70
            filtered_matches = []
            seen_asins = set()
            for match in matches:
                product_row = df[df["product_title"] == match[0]]
                if not product_row.empty:
                    asin = product_row["parent_asin"].iloc[0]
                    if asin not in seen_asins:
                        seen_asins.add(asin)
                        filtered_matches.append(match[0])

            if len(filtered_matches) > 1:
                options = "\n".join(
                    [f"{i + 1}. {title[:70] + '...' if len(title) > 70 else title}" for i, title in enumerate(filtered_matches)] 
                )
                dispatcher.utter_message(text=f"I found multiple matches. Please select one:\n{options}")
                return [SlotSet("possible_products", filtered_matches)]

            elif filtered_matches:
                product_title = filtered_matches[0]
                return self.generate_chart(df, product_title, dispatcher)

            dispatcher.utter_message(text="No matching products found. Please try again.")
            return []

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred: {str(e)}")
            return []

    def extract_product_name(self, user_query: str) -> str:
       
        query = re.sub(r"^(what is the behavior of|what is the trend of|rating history for)\s*", "", user_query, flags=re.IGNORECASE)

        return query.strip().lower()

    def generate_chart(self, df: pd.DataFrame, product_title: str, dispatcher: CollectingDispatcher) -> list:
        try:
            
            df_filtered = df[df["product_title"].str.contains(product_title, case=False, na=False)].copy()
            
            df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"], errors="coerce", unit='ms')
            df_filtered = df_filtered.dropna(subset=["timestamp", "rating"])
            df_filtered["year_month"] = df_filtered["timestamp"].dt.to_period("M")
            avg_ratings = df_filtered.groupby(df_filtered["timestamp"].dt.to_period("M"))["rating"].mean()

            
            plt.figure(figsize=(10, 5))
            plt.plot(avg_ratings.index.astype(str), avg_ratings, marker='o', linestyle='-', linewidth=2)

            
            plt.xlabel("Date (MM/YY)")
            plt.ylabel("Average Rating (1-5)")
            plt.title(f"Average Rating Behavior for {product_title}")
            plt.xticks(rotation=45)
            plt.grid()
            
            folder_path = r"C:\Projects\lucy_chatbot\ratingbehaviour"
            os.makedirs(folder_path, exist_ok=True)
            file_name = f"{product_title.replace(' ', '_')}_rating_behaviour.png"
            file_path = os.path.join(folder_path, file_name)
            plt.savefig(file_path)
            plt.close()
                      
            dispatcher.utter_message(text=f"Here is the rating behavior for '{product_title}'.")
            dispatcher.utter_message(image=file_path)
            return [SlotSet("possible_products", None)]
        
        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred while generating the chart: {str(e)}")
            return []
