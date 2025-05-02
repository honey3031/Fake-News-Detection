import streamlit as st
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# --- Model Parameters ---
VOCAB_SIZE = 5000
MAX_LEN = 20

# --- Text Preprocessing ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Dummy LSTM Model ---
model = Sequential()
model.add(Embedding(VOCAB_SIZE, 40, input_length=MAX_LEN))
model.add(LSTM(100))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Prediction Function ---
def predict_news(text):
    cleaned = clean_text(text)
    encoded = one_hot(cleaned, VOCAB_SIZE)
    padded = pad_sequences([encoded], maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded)[0]
    label = "FAKE" if np.argmax(pred) == 1 else "REAL"
    return label, pred

# --- Streamlit Page Config ---
st.set_page_config(page_title="Fake News Classifier", page_icon="🧠", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
<style>
body {
    background-color: #181818;
    color: white;
}
.stApp {
    background-color: #181818;
}
.big-title {
    font-size: 3em;
    font-weight: bold;
    background: -webkit-linear-gradient(45deg, #ff0000, #ff9900);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
.subtext {
    text-align: center;
    font-size: 1.2em;
    color: #bbbbbb;
    margin-bottom: 2rem;
}
.input-box {
    background-color: #222;
    color: white;
    border: 1px solid #555;
    border-radius: 8px;
    padding: 0.5rem;
    width: 100%;
}
.result-box {
    background-color: #202020;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.5em;
    font-weight: bold;
    margin-top: 2rem;
}
.real {
    color: #00e676;
}
.fake {
    color: #ff1744;
}
.footer {
    text-align: center;
    font-size: 0.85em;
    color: #777;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<div class="big-title">🧠 Fake News Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Enter a news headline to check if it\'s REAL or FAKE</div>', unsafe_allow_html=True)

# --- Input Section ---
user_input = st.text_input("Enter headline", placeholder="e.g. Elon Musk lands a Tesla on the moon 🚀")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a valid news headline.")
    else:
        label, confidence = predict_news(user_input)
        score = round(float(confidence[np.argmax(confidence)]) * 100, 2)

        result_class = "fake" if label == "FAKE" else "real"
        emoji = "❌" if label == "FAKE" else "✅"
        color = "#ff1744" if label == "FAKE" else "#00e676"

        st.markdown(f"""
        <div class="result-box {result_class}">
            {emoji} The news is <span class="{result_class}">{label}</span><br>
            <span style="font-size: 0.9em; color: #ccc;">Confidence: {score}%</span>
        </div>
        """, unsafe_allow_html=True)

        st.progress(score / 100)

        with st.expander("🔍 Advanced Prediction Details"):
            st.json({
                "Headline": user_input,
                "Cleaned": clean_text(user_input),
                "Confidence Scores": {
                    "REAL": round(float(confidence[0]), 4),
                    "FAKE": round(float(confidence[1]), 4)
                },
                "Prediction": label
            })

# --- Footer ---
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit and LSTM 
</div>
""", unsafe_allow_html=True)
