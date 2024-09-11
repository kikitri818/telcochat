import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from googletrans import Translator
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    df = pd.read_csv("hf://datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/bitext-telco-llm-chatbot-training-dataset.csv")
    return df

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
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")
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
        st.error(f"번역 중 오류 발생: {e}")
        return text

# RAG 함수
def rag(query, df, embedding_model, seq2seq_model, tokenizer):
    query_embedding = embedding_model.encode([query])
    df['embedding'] = df['instruction'].apply(lambda x: embedding_model.encode(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], query_embedding)[0][0])
    most_similar = df.nlargest(1, 'similarity')
    
    context = most_similar['instruction'].values[0]
    input_text = f"질문: {context}\n답변:"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = seq2seq_model.generate(**inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"])
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer = clean_text(answer)
    return answer

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
        answer = rag(query, df, embedding_model, seq2seq_model, tokenizer)
        st.write("답변:", answer)
