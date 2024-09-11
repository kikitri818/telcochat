import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# KoGPT 모델 및 토크나이저 로드
@st.cache_resource
def load_kogpt_model():
    model_name = "skt/kogpt2-base-v2"  # 한국어에 특화된 모델
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_kogpt_model()

# 응답 생성 함수
def generate_response(query):
    try:
        prompt = f"""
다음은 통신사 고객 상담원의 응답 예시입니다. 이를 참고하여 고객의 질문에 친절하고 자연스럽게 한국어로 답변해주세요.

고객: 요금제를 변경하고 싶어요.
상담원: 네, 고객님. 요금제 변경 문의 주셔서 감사합니다. 요금제 변경은 다음과 같은 방법으로 가능합니다:
1. 고객센터 전화: 114로 전화하셔서 상담원과 직접 통화하며 변경하실 수 있습니다.
2. 온라인/앱: 저희 통신사 홈페이지나 모바일 앱에 로그인하신 후 '요금제 변경' 메뉴를 이용하실 수 있습니다.
3. 대리점 방문: 가까운 통신사 대리점을 방문하셔서 직접 변경하실 수 있습니다.
변경하고자 하는 특정 요금제가 있으시다면 말씀해 주세요. 고객님의 사용 패턴에 맞는 최적의 요금제를 추천해 드리겠습니다.

고객: 인터넷 속도가 느린 것 같아요.
상담원: 네, 고객님. 인터넷 속도 문제로 불편을 겪고 계시는군요. 먼저 다음과 같은 간단한 해결 방법을 시도해 보시겠어요?
1. 모뎀/공유기 재시작: 전원을 껐다가 다시 켜보세요.
2. Wi-Fi 신호 확인: 공유기와의 거리를 좁혀보세요.
3. 다른 기기에서도 속도가 느린지 확인해 보세요.
이렇게 해도 개선되지 않는다면, 기술 지원팀의 원격 진단이 필요할 수 있습니다. 원격 진단을 원하시면 말씀해 주세요.

고객: {query}
상담원:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.95, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        answer = response.split("상담원:")[-1].strip()
        
        if not answer or len(answer) < 20 or "{{" in answer or "}}" in answer or any(word in answer.lower() for word in ["english", "provider"]):
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
