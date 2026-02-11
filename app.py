import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# ===============================
# Download NLTK Resources (Safe for Cloud)
# ===============================
@st.cache_resource
def load_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

load_nltk()

# ===============================
# Load Model + Vectorizer
# ===============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, vectorizer = load_model()

# ===============================
# Text Preprocessing
# ===============================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# ===============================
# Streamlit UI
# ===============================
st.title("🛒 Flipkart Product Review Sentiment Analysis")
st.write("Enter a product review to detect whether it is **Positive** or **Negative**.")

review = st.text_area("✍️ Enter Review Text")

# ===============================
# Prediction
# ===============================
if st.button("Predict Sentiment"):

    if model is None or vectorizer is None:
        st.error("Model not loaded. Check .pkl files and requirements.txt")
    elif review.strip() == "":
        st.warning("Please enter a review")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]

        if prediction == 1:
            st.success("✅ Positive Review 😊")
        else:
            st.error("❌ Negative Review 😞")