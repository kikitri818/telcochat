import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    dataset = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset", split="train")
    df = pd.DataFrame(dataset)
    return df

# Sentence Transformer 모델 로드
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# KoGPT 모델 및 토크나이저 로드
@st.cache_resource
def load_kogpt_model():
    model_name = "skt/kogpt2-base-v2"  # 한국어에 특화된 모델
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# 데이터 및 모델 로드
df = load_data()
sentence_model = load_sentence_transformer()
tokenizer, model = load_kogpt_model()

# 임베딩 생성
@st.cache_data
def create_embeddings(_df, _model):
    return _model.encode(_df['instruction'].tolist())

embeddings = create_embeddings(df, sentence_model)

# RAG 함수
def rag(query, top_k=3):
    try:
        query_embedding = sentence_model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        context = "\n".join(df.iloc[top_indices]['response'].tolist())
        
        prompt = f"""
고객님의 질문: {query}

위 질문에 대해 다음 정보를 참고하여 답변해주세요:
{context}

한국어로 자연스럽고 친절하게 답변해주세요. 고객 상담원이 대화하듯이, 불필요한 정보는 제외하고 핵심만 간결하게 말씀해 주세요.

답변:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        answer = response.split("답변:")[-1].strip()
        
        if not answer or len(answer) < 10:
            answer = "죄송합니다. 현재 답변을 생성할 수 없습니다. 다시 한 번 질문해 주시겠어요?"
        
        return answer
    except Exception as e:
        return f"죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다: {str(e)}"

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

user_input = st.text_input("질문을 입력하세요:")

if user_input:
    answer = rag(user_input)
    st.write("답변:", answer)
    
