import streamlit as st
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch
import pandas as pd

@st.cache_resource
def load_data():
    dataset = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset")
    df = pd.DataFrame(dataset['train'])
    st.write("데이터셋 구조:")
    st.write(df.head())
    st.write("열 이름:", df.columns.tolist())
    return df

@st.cache_resource
def prepare_data_and_index(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['instruction'])
    return vectorizer, tfidf_matrix

@st.cache_resource
def train_model(df):
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preprocess_function(examples):
        inputs = ["question: " + q for q in examples['instruction']]
        targets = examples['response']
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = df.to_dict(orient='list')
    dataset = preprocess_function(dataset)

    train_dataset = dataset
    
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
    )

    trainer.train()
    return model, tokenizer

st.title("텔코 고객센터 챗봇")

df = load_data()

if df is not None and 'instruction' in df.columns and 'response' in df.columns:
    vectorizer, tfidf_matrix = prepare_data_and_index(df)
    model, tokenizer = train_model(df)

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        # RAG: Retrieve similar questions
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-3:][::-1]
        
        context = " ".join(df.iloc[top_indices]['response'].tolist())
        
        # Generate response
        input_text = f"question: {user_input} context: {context}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
        outputs = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        st.write(f"챗봇 응답: {response}")
else:
    st.error("올바른 데이터를 불러오지 못했습니다. 데이터셋의 구조를 확인해주세요.")
    
