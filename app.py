import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from googletrans import Translator

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

# 번역기 초기화
translator = Translator()

# 텍스트 정제 함수
def clean_text(text):
    # 특수 태그들을 적절한 한글 표현으로 대체
    replacements = {
        r'\{\{CURRENT_PROVIDER\}\}': '현재 통신사',
        r'\{\{NEW_PROVIDER\}\}': '새로운 통신사',
        r'\{\{PROVIDER\}\}': '통신사',
        r'\{\{PRODUCT\}\}': '상품',
        r'\{\{SERVICE\}\}': '서비스',
        r'\{\{PORTING_CODE\}\}': '번호이동 인증번호',
        r'\{\{SUPPORT_TEAM_CONTACT\}\}': '고객 지원팀 연락처',
        # 추가 태그들에 대해서도 비슷하게 처리
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text

# 번역 함수
def translate_to_korean(text):
    try:
        return translator.translate(text, dest='ko').text
    except Exception as e:
        st.error(f"번역 중 오류 발생: {e}")
        return text

# RAG 함수
def rag(query, df, model):
    query_embedding = model.encode([query])
    df['embedding'] = df['instruction'].apply(lambda x: model.encode(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], query_embedding)[0][0])
    most_similar = df.nlargest(1, 'similarity')
    answer = most_similar['response'].values[0]
    answer = clean_text(answer)
    answer = translate_to_korean(answer)
    return answer

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

df = load_data()
model = load_model()

query = st.text_input("질문을 입력하세요:")

if query:
    answer = rag(query, df, model)
    st.write("답변:", answer)
