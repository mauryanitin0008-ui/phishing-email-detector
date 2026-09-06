import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️"
)

st.title("🛡️ Phishing Email Detector")
st.write("Enter an email below to check whether it is Phishing or Safe.")

# Load dataset
data = pd.read_csv(
    "emails.csv",
    header=None,
    names=["text", "label"]
)

# Prepare training data
X = data["text"].astype(str)
y = data["label"]

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Email input
email = st.text_area(
    "Enter Email Text:",
    height=200,
    placeholder="Paste the email text here..."
)

if st.button("🔍 Check Email"):
    if email.strip():
        email_vector = vectorizer.transform([email])
        result = model.predict(email_vector)[0]

        if str(result).lower() == "phishing":
            st.error("🚨 PHISHING EMAIL DETECTED")
        else:
            st.success("✅ SAFE EMAIL")
    else:
        st.warning("Please enter an email first.")
