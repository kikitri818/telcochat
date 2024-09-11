import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from googletrans import Translator
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import time
import concurrent.futures

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    df = pd.read_csv("hf://datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/bitext-telco-llm-chatbot-training-dataset.csv")
    return df.head(50)  # 데이터 크기를 50개로 제한

# 임베딩 모델 로드
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Fine-tuning을 위한 모델 및 토크나이저 로드
@st.cache_resource
def load_seq2seq_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "ko_KR"
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {str(e)}")
        tokenizer = None
        model = None
    return tokenizer, model

# 번역기 초기화
translator = Translator()

# 텍스트 정제 함수
def clean_text(text):
    replacements = {
        r'\{\{CURRENT_PROVIDER\}\}': '현재 통신사',
        r'\{\{NEW_PROVIDER\}\}': '새로운 통신사',
        r'\{\{PROVIDER\}\}': '통신사',
        r'\{\{PRODUCT\}\}': '상품',
        r'\{\{SERVICE\}\}': '서비스',
        r'\{\{PORTING_CODE\}\}': '번호이동 인증번호',
        r'\{\{SUPPORT_TEAM_CONTACT\}\}': '고객 지원팀 연락처',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text

# 번역 함수
def translate_to_korean(text):
    try:
        return translator.translate(text, dest='ko').text
    except Exception as e:
        return text

# 임베딩 계산 함수
@st.cache_data
def compute_embeddings(_df, _embedding_model):
    return [_embedding_model.encode(text).tolist() for text in _df['instruction']]

# RAG 함수
def rag(query, df, embedding_model, seq2seq_model, tokenizer):
    try:
        query_embedding = embedding_model.encode([query])
        
        if 'embedding' not in df.columns:
            df['embedding'] = compute_embeddings(df, embedding_model)
        
        df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], query_embedding)[0][0])
        most_similar = df.nlargest(1, 'similarity')
        
        context = most_similar['instruction'].values[0]
        input_text = f"질문: {context}\n답변:"
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = seq2seq_model.generate(**inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"])
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        answer = clean_text(answer)
        answer = translate_to_korean(answer)
        
        return answer
    except Exception as e:
        return f"답변 생성 중 오류 발생: {str(e)}"

# 시간 제한 있는 함수 실행
def run_with_timeout(func, args, timeout):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return "답변 생성 시간이 초과되었습니다."

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

df = load_data()
embedding_model = load_embedding_model()
tokenizer, seq2seq_model = load_seq2seq_model()

if tokenizer is None or seq2seq_model is None:
    st.error("모델 로딩에 실패했습니다. 앱을 다시 시작해주세요.")
else:
    query = st.text_input("질문을 입력하세요:")

    if query:
        with st.spinner('답변을 생성 중입니다...'):
            answer = run_with_timeout(rag, (query, df, embedding_model, seq2seq_model, tokenizer), timeout=30)
        st.write("답변:", answer)
