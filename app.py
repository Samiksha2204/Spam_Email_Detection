import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

try:
    with open('D:/Documents/TY/DMW/Project/vectorizer.pkl', 'rb') as vec_file:
        tfidf = pickle.load(vec_file, encoding='latin1')
    with open('D:/Documents/TY/DMW/Project/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file, encoding='latin1')
except Exception as e:
    st.error(f"Error occurred while loading the models: {e}")

# Inject custom CSS to change background color
st.markdown("""<style>
                 body {
                 background-color: #c0ffcc;
                 color: #ff69b4;
                 }
                 </style>""", unsafe_allow_html=True)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")