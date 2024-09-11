import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

@st.cache_resource
def load_data():
    df = pd.read_csv("https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/raw/main/bitext-telco-llm-chatbot-training-dataset.csv")
    return df.sample(n=100, random_state=42)

@st.cache_resource
def train_model(df):
    X = df['input']
    y = df['intent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(y.unique()))

    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
        eval_dataset=test_encodings,
    )

    trainer.train()
    return model, tokenizer, y.unique()

st.title("텔코 고객센터 챗봇")

df = load_data()
model, tokenizer, intents = train_model(df)

user_input = st.text_input("질문을 입력하세요:")

if user_input:
    input_encoding = tokenizer(user_input, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**input_encoding)
    predicted_intent = intents[output.logits.argmax().item()]
    
    st.write(f"예측된 의도: {predicted_intent}")
    # 여기에 의도에 따른 응답 로직을 추가할 수 있습니다.
    
