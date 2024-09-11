import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    dataset = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset", split="train")
    df = pd.DataFrame(dataset)
    if len(df) > 100:  # 데이터가 100개 이상이면 샘플링
        return df.sample(n=100, random_state=42)
    return df

df = load_data()

# 데이터프레임 구조 확인
st.write("데이터프레임 열:", df.columns)
st.write("데이터프레임 샘플:", df.head())

# Sentence Transformer 모델 로드
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

sentence_model = load_sentence_transformer()

# 임베딩 생성
@st.cache_data
def create_embeddings(_df, _model):
    return _model.encode(_df['instruction'].tolist())

embeddings = create_embeddings(df, sentence_model)

# KoGPT 모델 및 토크나이저 로드
@st.cache_resource
def load_kogpt_model():
    model_name = "skt/kogpt2-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_kogpt_model()

# Fine-tuning
def fine_tune_model(_df, _tokenizer, _model):
    dataset = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset", split="train")
    
    def tokenize_function(examples):
        return _tokenizer(examples["instruction"] + " " + examples["response"], truncation=True, padding="max_length", max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    return _model

model = fine_tune_model(df, tokenizer, model)

# RAG 함수
def rag(query, top_k=3):
    query_embedding = sentence_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    context = "\n".join(df.iloc[top_indices]['response'].tolist())
    
    prompt = f"다음 정보를 바탕으로 질문에 답변해주세요:\n\n{context}\n\n질문: {query}\n답변:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

user_input = st.text_input("질문을 입력하세요:")

if user_input:
    answer = rag(user_input)
    st.write("답변:", answer)
