import nltk

import datetime
import csv
import json
import random
import streamlit as st
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import pickle
import numpy as np
import os

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('omw-1.4')  # For better lemmatization support

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents and the trained model
def load_resources():
    try:
        with open('intents_2.json', 'r') as file:
            intents = json.load(file)
    except FileNotFoundError:
        print("Error: 'intents_2.json' file not found.")
        return None, None, None

    try:
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
    except FileNotFoundError:
        print("Error: 'words.pkl' or 'classes.pkl' file not found.")
        return None, None, None

    try:
        model = load_model('chatbot_simplilearnmodel.h5')
    except (OSError, IOError):
        print("Error: 'chatbot_simplilearnmodel.h5' model file not found.")
        return None, None, None

    return intents, words, classes, model

intents, words, classes, model = load_resources()

if intents is None or words is None or classes is None or model is None:
    print("Initialization failed. Please check the files.")
    exit()

# Preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return " ".join(sentence_words)

# Convert sentence into bag-of-words
def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict intent of user input
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.30  # Confidence threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if not results:
        return [{"intent": "fallback", "probability": "0"}]
    return [{"intent": classes[results[0][0]], "probability": str(results[0][1])}]

# Generate a response based on predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

# Initialize the chat history
def init_chat_history():
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

# Save a conversation to the chat history
def save_to_chat_history(user_input, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([user_input, response, timestamp])

# Streamlit application
def main():
    st.title("KrishiMitra:Chatbot with Intent Recognition")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Chat with our KrishiMitra that will help you to get all the answers related to farms")
        init_chat_history()

        user_input = st.text_input("You:")
        if user_input:
            cleaned_message = clean_up_sentence(user_input)
            intents_list = predict_class(user_input)

            if not intents_list or intents_list[0]["intent"] == "fallback":
                bot_response = "I'm sorry, I didn't understand that."
            else:
                bot_response = get_response(intents_list, intents)

            st.text_area("Chatbot:", value=bot_response, height=120, max_chars=None)

            save_to_chat_history(user_input, bot_response)

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.subheader("About")
        st.write("""
        This chatbot is built using NLP techniques and a Keras deep learning model. It classifies user input into intents and generates appropriate responses. 
        """)

if __name__ == "__main__":
    main()
