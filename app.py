import streamlit as st
import pickle

# Load the model and TfidfVectorizer from the pickle files
model = pickle.load(open('m1.pkl', 'rb'))
vectorizer = pickle.load(open('feature_extraction.pkl', 'rb'))

# Title of the Streamlit app
st.title('Email Spam Classifier')

# Input text box for user to enter an email message
message = st.text_area('Enter an email message:')

# Predict the category of the message when the button is clicked
if st.button('Predict'):
    if message:
        # Transform the input message using the loaded TfidfVectorizer
        transformed_message = vectorizer.transform([message])
        
        # Predict the category using the loaded model
        prediction = model.predict(transformed_message)
        
        # Display the predicted category
        if prediction == [1]:
            st.write('The message is: **Not Spam**')
        else:
            st.write('The message is: **Spam**')
    else:
        st.write('Cannot be classified.')
