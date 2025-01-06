import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("KrishiMitra: Chatbot with Intent Recognition")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Happy farming!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")

        # Read the conversation history from the CSV file
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            rows = list(csv_reader)

        # Display the conversation history
        for row in reversed(rows):  # Show the latest conversations first
            st.text(f"User: {row[0]}")
            st.text(f"Chatbot: {row[1]}")
            st.text(f"Timestamp: {row[2]}")
            st.markdown("---")

        # Allow user to refresh by setting a session state variable
        if 'refresh_history' not in st.session_state:
            st.session_state['refresh_history'] = False

        # Check if the user wants to refresh
        if st.button("Refresh History"):
            st.session_state['refresh_history'] = True
            st.experimental_rerun()  # Rerun the app to reload the history

        # Reset the session state variable if not refreshing
        if st.session_state['refresh_history']:
            st.session_state['refresh_history'] = False

    # About Menu
    elif choice == "About":
        st.write("**KrishiMitra: Your Farming Chatbot Companion**")
        st.subheader("Project Overview:")
        st.write("""
        KrishiMitra is an intelligent chatbot designed to assist farmers with their agricultural needs. 
        It leverages Natural Language Processing (NLP) and Machine Learning to recognize user intents and provide appropriate responses. 
        The chatbot is trained on a dataset containing a variety of intents relevant to farming, including crop selection, pest control, irrigation, weather updates, and more.
        """)
        
        st.subheader("Features:")
        st.write("""
        - **Crop Advice**: Get suggestions on what crops to grow based on season and location.
        - **Fertilizer Recommendations**: Learn about suitable fertilizers and soil enrichment techniques.
        - **Pest Control**: Discover organic and chemical methods for managing pests effectively.
        - **Weather Updates**: Receive tips for preparing your farm for various weather conditions.
        - **Irrigation Guidance**: Understand efficient water management and irrigation techniques.
        - **Market Prices**: Stay updated on current market rates for various crops.
        - **Government Schemes**: Explore farming subsidies, crop insurance, and loans provided by the government.
        - **Conversation History**: Review past conversations to track advice and insights provided by the chatbot.
        """)
        
        st.subheader("Technologies Used:")
        st.write("""
        - **NLP and Logistic Regression**: Used to classify intents from user inputs.
        - **TF-IDF Vectorizer**: Extracts features from text data to understand user queries.
        - **Streamlit**: Provides an interactive and user-friendly interface for the chatbot.
        - **CSV-based Conversation Log**: Saves and displays chat history for future reference.
        """)

        st.subheader("Future Enhancements:")
        st.write("""
        - Integration with real-time weather APIs and market rate databases.
        - Incorporation of voice-based input and responses.
        - Support for regional languages to cater to a diverse user base.
        - Advanced recommendations using deep learning techniques.
        """)

        st.subheader("Conclusion:")
        st.write("""
        KrishiMitra is a step forward in leveraging technology to empower farmers with knowledge and guidance. By providing quick and relevant answers, the chatbot aims to improve farming productivity and decision-making.
        """)

if __name__ == '__main__':
    main()
