import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_data
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    df = pd.read_csv(url)
    return df.sample(n=1000, random_state=42)  # 1000개 샘플만 사용

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "skt/kogpt2-base-v2"  # 한국어 GPT-2 모델
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_data
def precompute_embeddings(_df, _sentence_transformer):
    return _df.assign(embedding=_df['instruction'].apply(lambda x: _sentence_transformer.encode(x).tolist()))

def retrieve_relevant_context(query, df, sentence_transformer):
    query_embedding = sentence_transformer.encode([query])
    similarities = cosine_similarity(query_embedding, np.stack(df['embedding'].values))
    df['similarity'] = similarities[0]
    return df.nlargest(3, 'similarity')

def generate_korean_response(context, query, tokenizer, model):
    input_text = f"질문: {query}\n컨텍스트: {context}\n답변:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("답변:")[-1].strip()

def main():
    st.title("텔코 챗봇")

    df = load_data()
    sentence_transformer = load_sentence_transformer()
    tokenizer, model = load_model_and_tokenizer()
    
    df = precompute_embeddings(df, sentence_transformer)

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        relevant_context = retrieve_relevant_context(user_input, df, sentence_transformer)
        
        if not relevant_context.empty:
            context = relevant_context.iloc[0]['instruction'] + " " + relevant_context.iloc[0]['response']
            response = generate_korean_response(context, user_input, tokenizer, model)
            source = "Fine-tuning 데이터"
        else:
            response = "죄송합니다. 해당 질문에 대한 정확한 답변을 찾지 못했습니다."
            source = "기본 응답"

        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}를 참고했습니다.")

if __name__ == "__main__":
    main()
