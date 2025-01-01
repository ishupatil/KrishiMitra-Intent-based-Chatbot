#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import streamlit as st


lemmatizer = WordNetLemmatizer()

# Load data and model
with open('intents_2.json', 'r') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_simplilearnmodel.h5')

# Extract all patterns from intents for strict matching
all_patterns = []
for intent in intents['intents']:
    all_patterns.extend(intent['patterns'])

# Clean up the input sentence
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

# Predict the class of the input
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.30  # Set a threshold for confidence
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if not results:  # No results above the threshold
        return [{"intent": "fallback", "probability": "0"}]
    return [{"intent": classes[results[0][0]], "probability": str(results[0][1])}]

# Get response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."

# Main loop
print("Chatbot is running!")
while True:
    try:
        message = input("You: ").strip()
        if not message:  # Check for empty input
            print("Bot: Please enter a valid input.")
            continue

        # Preprocess user input for strict matching
        cleaned_message = clean_up_sentence(message)

        # Check if input matches any pattern
        if not any(cleaned_message == clean_up_sentence(pattern) for pattern in all_patterns):
            print("Bot: You are inputting the wrong input. Please try asking something else.")
            continue

        # Get intents and validate
        ints = predict_class(message)
        if not ints:  # No intents above the threshold
            print("Bot: You are inputting the wrong input. Please try asking something else.")
        else:
            res = get_response(ints, intents)
            print(f"Bot: {res}")
    except Exception as e:
        print(f"Bot: An error occurred: {e}")



# In[ ]:





# In[ ]:




