import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize the vectorizer (ensure you use the same one used during training)
with open('tfidf_vectorizer.pkl', 'rb') as vector_file:
    vectorizer = pickle.load(vector_file) 

# Define a function to predict sentiment
def predict_sentiment(text):
    # Transform the text to the same format used during training
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction

# Create the Streamlit app layout
st.title('Sentiment Analysis App')
user_input = st.text_area("Enter a sentence:")

if st.button('Predict'):
    if user_input:
        prediction = predict_sentiment(user_input)
        if prediction[0]==0:
            st.write("The sentiment is: **Negative**")
        elif prediction[0]==1:
            st.write("The sentiment is: **neutral**")
        else:
            st.write("The sentiment is: **Positive**")
        


        
    else:
        st.write("Please enter a sentence to predict.")
