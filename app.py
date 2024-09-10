import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from deep_translator import GoogleTranslator
import re

@st.cache_data
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    return pd.read_csv(url)

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

@st.cache_resource
def load_translator():
    return GoogleTranslator(source='en', target='ko')

@st.cache_data
def precompute_embeddings(_df, _sentence_transformer):
    return _df.assign(embedding=_df['instruction'].apply(lambda x: _sentence_transformer.encode(x).tolist()))

def retrieve_relevant_context(query, df, sentence_transformer):
    query_embedding = sentence_transformer.encode([query])
    similarities = cosine_similarity(query_embedding, np.stack(df['embedding'].values))
    df['similarity'] = similarities[0]
    return df.nlargest(3, 'similarity')

def clean_response(response):
    template_mapping = {
        'WEBSITE_URL': '웹사이트',
        'INVOICE_SECTION': '청구서 섹션',
        'DISPUTE_INVOICE_OPTION': '청구서 분쟁 옵션',
        'SUPPORT_TEAM_CONTACT': '고객 지원팀 연락처',
        'DAYS_NUMBER': '며칠',
        'TOTAL_AMOUNT': '총 금액',
        'ACCOUNT_SECTION': '계정 섹션',
        'PAYMENT_METHOD': '결제 방법',
        'CUSTOMER_SERVICE': '고객 서비스',
        'ACCOUNT_DETAILS': '계정 세부 정보',
        'BILL_AMOUNT': '청구 금액',
        'DUE_DATE': '납부 기한',
        'PAYMENT_OPTIONS': '결제 옵션',
        'BILLING_CYCLE': '청구 주기',
        'ACCOUNT_NUMBER': '계정 번호',
        'SERVICE_PLAN': '서비스 플랜',
        'DATA_USAGE': '데이터 사용량',
        'CALL_MINUTES': '통화 시간',
        'TEXT_MESSAGES': '문자 메시지 수',
        'SUPPORT_HOURS': '고객 지원 시간',
        'CONTACT_NUMBER': '연락처 번호',
        'ACTIVATION_SECTION': '활성화 섹션',
        'PRODUCT_NAME': '제품명',
        'SUBSCRIPTION_DETAILS': '구독 정보',
        'PACKAGE_NAME': '패키지명',
        'CANCELLATION_POLICY': '해지 정책',
        'REFUND_POLICY': '환불 정책',
        'TERMS_AND_CONDITIONS': '이용 약관',
        'PRIVACY_POLICY': '개인정보 처리방침'
    }
    
    def replace_template(match):
        key = match.group(1)
        return template_mapping.get(key, '')
    
    # 모든 템플릿 변수를 대체하거나 제거
    cleaned_response = re.sub(r'\{\{(\w+)\}\}', replace_template, response)
    
    return cleaned_response.strip()

def generate_response(query, relevant_context, translator):
    if not relevant_context.empty and relevant_context.iloc[0]['similarity'] > 0.5:
        responses = relevant_context['response'].apply(clean_response).tolist()
        combined_response = ' '.join(responses)
        translated_response = translator.translate(combined_response)
        return translated_response, "RAG/Fine-tuning"
    else:
        return "죄송합니다. 해당 질문에 대한 정확한 답변을 찾지 못했습니다.", "검색 결과 없음"

def main():
    st.title("텔코 챗봇")

    df = load_data()
    sentence_transformer = load_sentence_transformer()
    translator = load_translator()
    df = precompute_embeddings(df, sentence_transformer)

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        relevant_context = retrieve_relevant_context(user_input, df, sentence_transformer)
        response, source = generate_response(user_input, relevant_context, translator)
        
        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}을 통해 생성되었습니다.")

if __name__ == "__main__":
    main()
