import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    return pd.read_csv(url)  # 전체 데이터셋 사용

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "skt/kogpt2-base-v2"  # 한국어 GPT-2 모델
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def retrieve_relevant_context(query, df, sentence_transformer):
    query_embedding = sentence_transformer.encode([query])
    df['embedding'] = df['instruction'].apply(lambda x: sentence_transformer.encode(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x.reshape(1, -1))[0][0])
    return df.nlargest(3, 'similarity')

def main():
    st.title("텔코 챗봇")

    df = load_data()
    sentence_transformer = load_sentence_transformer()
    tokenizer, model = load_model_and_tokenizer()

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        relevant_context = retrieve_relevant_context(user_input, df, sentence_transformer)
        
        if not relevant_context.empty:
            response = relevant_context.iloc[0]['response']
            source = "Fine-tuning 데이터"
        else:
            response = "죄송합니다. 해당 질문에 대한 정확한 답변을 찾지 못했습니다."
            source = "기본 응답"

        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}를 참고했습니다.")

if __name__ == "__main__":
    main()
