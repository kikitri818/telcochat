import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    dataset = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset", split="train")
    df = pd.DataFrame(dataset)
    if len(df) > 1000:  # 데이터가 1000개 이상이면 샘플링
        return df.sample(n=1000, random_state=42)
    return df

# KoGPT 모델 및 토크나이저 로드
@st.cache_resource
def load_kogpt_model():
    model_name = "skt/kogpt2-base-v2"  # 한국어에 특화된 모델
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# 데이터 및 모델 로드
df = load_data()
tokenizer, model = load_kogpt_model()

# 응답 생성 함수
def generate_response(query):
    try:
        prompt = f"""
다음은 통신사 고객 상담원의 응답 예시입니다. 이를 참고하여 고객의 질문에 친절하고 자연스럽게 한국어로 답변해주세요.

고객: 요금제를 변경하고 싶어요.
상담원: 네, 고객님. 요금제 변경에 대해 문의주셨군요. 요금제 변경은 간단한 절차로 가능합니다. 현재 사용 중인 요금제와 변경하고자 하는 요금제를 알려주시면 상세히 안내해 드리겠습니다. 혹시 특정 요금제를 고려하고 계신가요?

고객: 인터넷 속도가 느린 것 같아요.
상담원: 네, 고객님. 인터넷 속도 문제로 불편을 겪고 계시는군요. 먼저 현재 사용 중인 인터넷 상품과 대략적인 위치를 알려주시면, 해당 지역의 네트워크 상태를 확인해보겠습니다. 또한, 간단한 해결 방법을 몇 가지 안내해 드릴 수 있습니다. 함께 확인해 보시겠어요?

고객: {query}
상담원:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        answer = response.split("상담원:")[-1].strip()
        
        if not answer or len(answer) < 10 or "{{" in answer or "}}" in answer or any(word in answer.lower() for word in ["english", "provider"]):
            answer = "죄송합니다. 현재 시스템 점검 중입니다. 고객님의 문의사항에 대해 정확한 답변을 드리기 위해 고객센터(1234-5678)로 연락 주시면 상세히 안내해 드리겠습니다. 불편을 드려 죄송합니다."
        
        return answer
    except Exception as e:
        return f"죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다. 고객센터(1234-5678)로 문의 부탁드립니다."

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

user_input = st.text_input("질문을 입력하세요:")

if user_input:
    answer = generate_response(user_input)
    st.write("답변:", answer)
