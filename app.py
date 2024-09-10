import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from deep_translator import GoogleTranslator
import requests
from bs4 import BeautifulSoup

@st.cache_data
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    df = pd.read_csv(url)
    return df.sample(n=1000, random_state=42)  # 1000개 샘플만 사용

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

@st.cache_resource
def load_translator():
    return GoogleTranslator(source='auto', target='ko')

@st.cache_data
def precompute_embeddings(_df, _sentence_transformer):
    return _df.assign(embedding=_df['instruction'].apply(lambda x: _sentence_transformer.encode(x).tolist()))

def retrieve_relevant_context(query, df, sentence_transformer):
    query_embedding = sentence_transformer.encode([query])
    similarities = cosine_similarity(query_embedding, np.stack(df['embedding'].values))
    df['similarity'] = similarities[0]
    return df.nlargest(3, 'similarity')

def translate_text(text, translator, source='en', target='ko'):
    try:
        translator.source = source
        translator.target = target
        return translator.translate(text)
    except:
        return f"번역 중 오류가 발생했습니다. 원본 텍스트: {text}"

def search_tworld(query):
    url = "https://www.tworld.co.kr/normal.do?serviceId=S_PROD2001&viewId=V_PROD2001&prod_id=NA00007395"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    results = []
    keywords = query.lower().split()
    for elem in soup.find_all(['p', 'li', 'h3', 'h4']):
        text = elem.text.strip().lower()
        if any(keyword in text for keyword in keywords):
            results.append(elem.text.strip())
    
    return results

def web_search(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    data = response.json()
    
    results = []
    if data.get('Abstract'):
        results.append(data['Abstract'])
    if data.get('RelatedTopics'):
        for topic in data['RelatedTopics'][:2]:  # 최대 2개의 관련 토픽 추가
            if 'Text' in topic:
                results.append(topic['Text'])
    
    return results if results else [f"'{query}'에 대한 정보를 찾지 못했습니다."]

def perform_search(query, df, sentence_transformer, translator):
    # 1. Fine-tuning 데이터 검색
    relevant_context = retrieve_relevant_context(query, df, sentence_transformer)
    if not relevant_context.empty and relevant_context.iloc[0]['similarity'] > 0.7:
        response = relevant_context.iloc[0]['response']
        return translate_text(response, translator), "Fine-tuning 데이터"
    
    # 2. T world 웹사이트 검색
    tworld_results = search_tworld(query)
    if tworld_results:
        response = " ".join(tworld_results[:3])  # 최대 3개의 결과만 사용
        return response, "T world 웹사이트"
    
    # 3. 일반 웹 검색
    web_results = web_search(query)
    if web_results:
        response = " ".join(web_results[:3])  # 최대 3개의 결과만 사용
        return response, "웹 검색"
    
    return "죄송합니다. 해당 질문에 대한 정확한 답변을 찾지 못했습니다.", "기본 응답"

def main():
    st.title("텔코 챗봇")

    df = load_data()
    sentence_transformer = load_sentence_transformer()
    translator = load_translator()
    
    df = precompute_embeddings(df, sentence_transformer)

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        response, source = perform_search(user_input, df, sentence_transformer, translator)
        
        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}를 참고했습니다.")

if __name__ == "__main__":
    main()
