# KrishiMitra:Intent Classification Chatbot with Web Application  

This repository features an **AI-powered chatbot** built using Python, Keras, and NLTK. The chatbot classifies user inputs into predefined intents and generates appropriate responses. It includes both a Command Line Interface (CLI) and an interactive web application built with Streamlit. 

## Features  

- **Intent Classification**: Classifies user messages into predefined intents using a Keras-based neural network.  
- **Natural Language Processing (NLP)**: Processes user input with tokenization, lemmatization, and text normalization using NLTK.  
- **Interactive Web Interface**: A GUI built with Streamlit for real-time chatbot interaction.  
- **Customizable Responses**: Easily extend intents and responses stored in a JSON file.  
- **Model Persistence**: Save and reuse the trained model using Pickle for efficient execution.  
## Installation  

1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/ishupatil/KrishiMitra-Intent-based-Chatbot.git  
   cd  KrishiMitra-Intent-based-Chatbot
   ```  

2. **Install dependencies**:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Run in CLI mode**:  
   ```bash  
   python Chatbot.py  
   ```  

4. **Run the web application**:  
   ```bash  
   streamlit run app.py  
   ```  

---

## How It Works  

1. **Input Preprocessing**: User messages are preprocessed using NLP techniques (tokenization, lemmatization).  
2. **Intent Prediction**: The processed text is passed to a Keras model trained to classify intents.  
3. **Response Generation**: A predefined response is fetched from the `intents_2.json` file based on the predicted intent.  

---

## File Structure  

- `Chatbot.py`: Script for the command-line chatbot.  
- `app.py`: Streamlit-based graphical chatbot interface.  
- `intents_2.json`: JSON file storing intents, patterns, and responses.  
- `new.py`: Serialized trained model file.  
- `requirements.txt`: List of dependencies.
- `words.pkl`:List of unique words or tokens derived from the training data (e.g., intents in intents_2.json).
- `vectorizer.pkl`: A CountVectorizer, TfidfVectorizer, or a custom vectorizer used to convert text data into numerical features.
- ## Web Application  

The chatbot’s web application provides a user-friendly interface for real-time interaction. Built with **Streamlit**, it can be run locally or deployed on cloud platforms.  

To launch the web app:  
```bash  
streamlit run app.py  
```  
## Contributions  
We welcome contributions! Fork this repository, make improvements, and open a pull request.  

