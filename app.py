import streamlit as st
import pickle
import nltk
import re

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

with open('models/nlp_model.pkl','rb') as file:
    saved_data=pickle.load(file)

model=saved_data['model']
vec=saved_data['tfidf']

tk=TweetTokenizer()
stop=stopwords.words('english')
snow=SnowballStemmer('english')

def preprocess_text(tweets):
    tweets = re.sub('[^a-zA-Z0-9]', ' ', tweets)
    words=tk.tokenize(tweets)
    words=[w for w in words if len(w)>=3]
    words=[snow.stem(i.lower()) for i in words]
    words=[w for w in words if w not in stop]

    return ' '.join(words)


st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0B0F19;
    color: #F8FAFC;
}

/* Main heading */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 1px;
    margin-top: 30px;
    margin-bottom: 5px;
}

/* Blue accent */
.main-title span {
    color: #3B82F6;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #94A3B8;
    margin-bottom: 40px;
}

/* Text input */
.stTextArea textarea {
    background-color: #151C2C;
    color: #FFFFFF;
    border: 1px solid #26364F;
    border-radius: 10px;
}

/* Button */
.stButton button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 25px;
    font-weight: 600;
}

.stButton button:hover {
    background-color: #1D4ED8;
    color: white;
    border: none;
}

/* Result styling */
.result-box {
    background-color: #151C2C;
    border: 1px solid #26364F;
    border-left: 4px solid #3B82F6;
    padding: 18px;
    border-radius: 10px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 class="main-title">
    <span class="icon">🍴</span>
    Dine<span>Insight</span>
</h1>

<p class="subtitle">
    Restaurant Review Sentiment Analysis
</p>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("Analyze a Restaurant Review")

review = st.text_area(
    "Enter your review",
    placeholder="Write your restaurant experience here...",
    height=150
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        preprocessed_text = preprocess_text(review)

        numerical = vec.transform([preprocessed_text])

        prediction = model.predict(numerical)

        if prediction[0] == 0:
            st.markdown("""
            <div class="result-box">
                <h3 style="color:#EF4444;">Negative Sentiment</h3>
                <p style="color:#94A3B8;">Oops! This review reflects a negative experience.</p>
            </div>
            """, unsafe_allow_html=True)

        elif prediction[0] == 1:
            st.markdown("""
            <div class="result-box">
                <h3 style="color:#22C55E;">Positive Sentiment</h3>
                <p style="color:#94A3B8;">Yay! This review reflects a positive experience.</p>
            </div>
            """, unsafe_allow_html=True)