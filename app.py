import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

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

# 텍스트 정제 함수
def clean_text(text):
    # {{CURRENT_PROVIDER}}, {{NEW_PROVIDER}} 등의 패턴을 적절한 한글 표현으로 대체
    text = re.sub(r'\{\{CURRENT_PROVIDER\}\}', '현재 통신사', text)
    text = re.sub(r'\{\{NEW_PROVIDER\}\}', '새로운 통신사', text)
    text = re.sub(r'\{\{PROVIDER\}\}', '통신사', text)
    text = re.sub(r'\{\{PRODUCT\}\}', '상품', text)
    text = re.sub(r'\{\{SERVICE\}\}', '서비스', text)
    # 추가적인 패턴들에 대해서도 비슷하게 처리
    return text

# RAG 함수
def rag(query, df, model):
    query_embedding = model.encode([query])
    df['embedding'] = df['instruction'].apply(lambda x: model.encode(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], query_embedding)[0][0])
    most_similar = df.nlargest(1, 'similarity')
    return clean_text(most_similar['response'].values[0])

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

df = load_data()
model = load_model()

query = st.text_input("질문을 입력하세요:")

if query:
    answer = rag(query, df, model)
    st.write("답변:", answer)
