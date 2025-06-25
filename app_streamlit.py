# app.py
import streamlit as st
import torch
import numpy as np
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import pandas as pd
import random

# Setup
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

# Load intents
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten and encode
df = pd.DataFrame([
    {"intent": intent["tag"], "pattern": p, "response": r}
    for intent in data["intents"]
    for p in intent["patterns"]
    for r in intent["responses"]
])

label_encoder = LabelEncoder()
df['intent_encoded'] = label_encoder.fit_transform(df['intent'])

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    stemmed = stemmer.stem(' '.join(words))
    return stemmed.strip()

# Load model & tokenizer
MODEL_PATH = "athayary/indobert"  # ganti kalau dari lokal
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

MAX_LEN = 128

def encode_texts(texts):
    encoded_batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return encoded_batch['input_ids'].to(device), encoded_batch['attention_mask'].to(device)

# Prediction
def predict_intent(text, threshold=0.1):
    cleaned = preprocess_text(text)
    input_ids, attn_mask = encode_texts([cleaned])

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        confidence = confidence.item()

        if confidence < threshold:
            return None, "Maaf, saya tidak memahami maksud Anda. Bisa dijelaskan lagi?"
        intent = label_encoder.inverse_transform([pred_class.item()])[0]
        responses = df[df['intent'] == intent]['response'].values
        return intent, random.choice(responses)

# Streamlit UI
# Streamlit UI (chat interface)
st.set_page_config(page_title="TanyaRasa Chatbot", page_icon="🤖")
st.title("🤖 TanyaRasa - Chatbot Bahasa Indonesia")

# Inisialisasi session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Form input pengguna
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Tanyakan sesuatu:", placeholder="Ketik pertanyaan kamu di sini...")
    submitted = st.form_submit_button("Kirim")

# Proses input jika dikirim
if submitted and user_input:
    intent, response = predict_intent(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# Tombol hapus riwayat
if st.button("🗑️ Hapus Riwayat Chat"):
    st.session_state.chat_history = []
    st.rerun()



# Tampilkan riwayat chat sebagai bubble
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"""
        <div style='text-align: right; background-color: #dcf8c6; padding: 10px; border-radius: 10px; margin: 5px 0;'>
            <b>Anda:</b> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='text-align: left; background-color: #f1f0f0; padding: 10px; border-radius: 10px; margin: 5px 0;'>
            <b>Chatbot:</b> {message}
        </div>
        """, unsafe_allow_html=True)

