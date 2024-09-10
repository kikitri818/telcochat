import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from googletrans import Translator

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
    return Translator()

@st.cache_data
def precompute_embeddings(_df, _sentence_transformer):
    return _df.assign(embedding=_df['instruction'].apply(lambda x: _sentence_transformer.encode(x).tolist()))

def retrieve_relevant_context(query, df, sentence_transformer):
    query_embedding = sentence_transformer.encode([query])
    similarities = cosine_similarity(query_embedding, np.stack(df['embedding'].values))
    df['similarity'] = similarities[0]
    return df.nlargest(3, 'similarity')

def translate_response(response, translator):
    try:
        translated = translator.translate(response, src='en', dest='ko')
        return translated.text
    except:
        return "번역 중 오류가 발생했습니다. 원본 응답: " + response

def main():
    st.title("텔코 챗봇")

    df = load_data()
    sentence_transformer = load_sentence_transformer()
    translator = load_translator()
    
    df = precompute_embeddings(df, sentence_transformer)

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        # 사용자 입력을 영어로 번역
        translated_input = translator.translate(user_input, src='ko', dest='en').text
        
        relevant_context = retrieve_relevant_context(translated_input, df, sentence_transformer)
        
        if not relevant_context.empty:
            response = relevant_context.iloc[0]['response']
            translated_response = translate_response(response, translator)
            source = "Fine-tuning 데이터"
        else:
            translated_response = "죄송합니다. 해당 질문에 대한 정확한 답변을 찾지 못했습니다."
            source = "기본 응답"

        st.write("챗봇 응답:")
        st.write(translated_response)
        st.write(f"위 답변은 {source}를 참고했습니다.")

if __name__ == "__main__":
    main()
