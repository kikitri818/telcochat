import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

@st.cache_data
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    df = pd.read_csv(url)
    return df.sample(n=1000, random_state=42)  # 1000개 샘플만 사용

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

@st.cache_data
def precompute_embeddings(_df, _sentence_transformer):
    return _df.assign(embedding=_df['instruction'].apply(lambda x: _sentence_transformer.encode(x).tolist()))

def retrieve_relevant_context(query, df, sentence_transformer):
    query_embedding = sentence_transformer.encode([query])
    similarities = cosine_similarity(query_embedding, np.stack(df['embedding'].values))
    df['similarity'] = similarities[0]
    return df.nlargest(3, 'similarity')

def web_search(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    data = response.json()
    results = []
    if data.get('Abstract'):
        results.append(data['Abstract'])
    for topic in data.get('RelatedTopics', []):
        if isinstance(topic, dict) and 'Text' in topic:
            results.append(topic['Text'])
    return results[:3]  # 최대 3개의 결과만 반환

def generate_response(query, fine_tuning_results, web_results):
    response = f"'{query}'에 대한 검색 결과입니다:\n\n"
    
    if not fine_tuning_results.empty and fine_tuning_results.iloc[0]['similarity'] > 0.7:
        response += "Fine-tuning 데이터 검색 결과:\n"
        response += fine_tuning_results.iloc[0]['response'] + "\n\n"
    elif web_results:
        response += "웹 검색 결과:\n"
        for i, result in enumerate(web_results, 1):
            response += f"{i}. {result}\n\n"
    else:
        response = f"죄송합니다. '{query}'에 대한 정보를 찾지 못했습니다. 다른 방식으로 질문을 해보시거나, 고객센터에 문의해 주세요."
    
    return response

def main():
    st.title("텔코 챗봇")

    df = load_data()
    sentence_transformer = load_sentence_transformer()
    df = precompute_embeddings(df, sentence_transformer)

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        fine_tuning_results = retrieve_relevant_context(user_input, df, sentence_transformer)
        web_results = web_search(user_input)
        response = generate_response(user_input, fine_tuning_results, web_results)
        
        st.write("챗봇 응답:")
        st.write(response)
        if not fine_tuning_results.empty and fine_tuning_results.iloc[0]['similarity'] > 0.7:
            st.write("위 답변은 Fine-tuning 데이터를 참고했습니다.")
        elif web_results:
            st.write("위 답변은 웹 검색을 참고했습니다.")
        else:
            st.write("검색 결과가 없습니다.")

if __name__ == "__main__":
    main()
