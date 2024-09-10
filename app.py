import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from deep_translator import GoogleTranslator

@st.cache_data
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    return pd.read_csv(url)  # 전체 데이터 사용

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "gpt2"  # 영어 모델 사용
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

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

def generate_response(query, context, tokenizer, model):
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    max_new_tokens = 150
    
    output = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

def perform_rag(query, df, sentence_transformer, tokenizer, model, translator):
    relevant_context = retrieve_relevant_context(query, df, sentence_transformer)
    if not relevant_context.empty:
        context = relevant_context.iloc[0]['instruction'] + " " + relevant_context.iloc[0]['response']
        response = generate_response(query, context, tokenizer, model)
        translated_response = translator.translate(response)
        return translated_response, "RAG/Fine-tuning"
    return "죄송합니다. 해당 질문에 대한 정확한 답변을 찾지 못했습니다.", "검색 결과 없음"

def main():
    st.title("텔코 챗봇")

    df = load_data()
    sentence_transformer = load_sentence_transformer()
    tokenizer, model = load_model_and_tokenizer()
    translator = load_translator()
    df = precompute_embeddings(df, sentence_transformer)

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        translated_query = translator.translate(user_input, source='ko', target='en')
        response, source = perform_rag(translated_query, df, sentence_transformer, tokenizer, model, translator)
        
        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}을 통해 생성되었습니다.")

if __name__ == "__main__":
    main()
