import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

@st.cache_resource
def load_data():
    df = pd.read_csv("https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/raw/main/bitext-telco-llm-chatbot-training-dataset.csv")
    return df

@st.cache_resource
def prepare_data(df):
    # 데이터셋의 열 이름을 출력하여 확인
    print(df.columns)
    X = df[df.columns[0]]  # 첫 번째 열을 입력으로 사용
    y = df[df.columns[1]]  # 두 번째 열을 레이블로 사용
    return train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model(X_train, X_test, y_train, y_test):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y_train)))

    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

    train_dataset = list(zip(train_encodings['input_ids'], train_encodings['attention_mask'], y_train))
    test_dataset = list(zip(test_encodings['input_ids'], test_encodings['attention_mask'], y_test))

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
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    return model, tokenizer, list(set(y_train))

st.title("텔코 고객센터 챗봇")

df = load_data()
X_train, X_test, y_train, y_test = prepare_data(df)
model, tokenizer, intents = train_model(X_train, X_test, y_train, y_test)

user_input = st.text_input("질문을 입력하세요:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_intent = intents[outputs.logits.argmax().item()]
    
    st.write(f"예측된 의도: {predicted_intent}")
