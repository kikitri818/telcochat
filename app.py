import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics.pairwise import cosine_similarity

# (이전 코드는 동일하게 유지)

# RAG 함수 수정
def rag(query, top_k=3):
    try:
        query_embedding = sentence_model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        context = "\n".join(df.iloc[top_indices]['response'].tolist())
        
        prompt = f"""다음은 고객 질문에 대한 관련 정보입니다:

{context}

고객 질문: {query}

위 정보를 바탕으로 고객 질문에 대해 한국어로 자연스럽게 답변해주세요. 
마치 고객 상담원이 친절하게 대화하듯이 답변을 작성해주세요. 
불필요한 정보는 제외하고 핵심만 간결하게 말씀해 주세요:

답변: """
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        answer = response.split("답변:")[-1].strip()
        
        return answer
    except Exception as e:
        return f"죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다: {str(e)}"

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

user_input = st.text_input("질문을 입력하세요:")

if user_input:
    answer = rag(user_input)
    st.write("답변:", answer)
