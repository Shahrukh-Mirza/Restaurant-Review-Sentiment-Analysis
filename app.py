import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords")

model = joblib.load("sentiment_model.pkl")
cv = joblib.load("count_vectorizer.pkl")

ps = PorterStemmer()

def preprocess(text):

    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()

    all_stopwords = stopwords.words('english')

    if 'not' in all_stopwords:
        all_stopwords.remove('not')

    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]

    review = ' '.join(review)

    return review


st.title("🍽 Restaurant Review Sentiment Analysis")

st.write("Enter a restaurant review below.")

review = st.text_area("Review")

if st.button("Predict"):

    cleaned_review = preprocess(review)

    vector = cv.transform([cleaned_review]).toarray()

    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.success("😊 Positive Review")

    else:
        st.error("😞 Negative Review")