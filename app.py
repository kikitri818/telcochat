import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    df = pd.read_csv("hf://datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/bitext-telco-llm-chatbot-training-dataset.csv")
    df = df.sample(n=100, random_state=42)  # 100개 샘플 추출
    return df

# 임베딩 모델 로드
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# RAG 함수
def rag(query, df, model):
    query_embedding = model.encode([query])
    df['embedding'] = df['instruction'].apply(lambda x: model.encode(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], query_embedding)[0][0])
    most_similar = df.nlargest(1, 'similarity')
    return most_similar['response'].values[0]

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

df = load_data()
model = load_model()

query = st.text_input("질문을 입력하세요:")

if query:
    answer = rag(query, df, model)
    st.write("답변:", answer)
