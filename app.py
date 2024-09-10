import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from deep_translator import GoogleTranslator

@st.cache_data
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    return pd.read_csv(url)

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def load_translator():
    return GoogleTranslator(source='en', target='ko')

def precompute_embeddings(df, sentence_transformer):
    return df.assign(embedding=df['instruction'].apply(lambda x: sentence_transformer.encode(x)))

def retrieve_relevant_context(query, df, sentence_transformer):
    query_embedding = sentence_transformer.encode([query])
    similarities = cosine_similarity(query_embedding, np.stack(df['embedding'].values))
    df['similarity'] = similarities[0]
    return df.nlargest(3, 'similarity')

def generate_response(query, relevant_context, translator):
    if not relevant_context.empty and relevant_context.iloc[0]['similarity'] > 0.5:
        response = relevant_context.iloc[0]['response']
        translated_response = translator.translate(response)
        return translated_response, "RAG/Fine-tuning"
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
