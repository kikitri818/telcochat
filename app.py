import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    df = pd.read_csv("hf://datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/bitext-telco-llm-chatbot-training-dataset.csv")
    return df

# 임베딩 모델 로드
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Fine-tuning을 위한 모델 및 토크나이저 로드
@st.cache_resource
def load_seq2seq_model():
    model_name = "google/mt5-small"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        tokenizer = None
        model = None
    return tokenizer, model

# 번역기 초기화
translator = Translator()

# 텍스트 정제 함수
def clean_text(text):
    replacements = {
        r'\{\{CURRENT_PROVIDER\}\}': '현재 통신사',
        r'\{\{NEW_PROVIDER\}\}': '새로운 통신사',
        r'\{\{PROVIDER\}\}': '통신사',
        r'\{\{PRODUCT\}\}': '상품',
        r'\{\{SERVICE\}\}': '서비스',
        r'\{\{PORTING_CODE\}\}': '번호이동 인증번호',
        r'\{\{SUPPORT_TEAM_CONTACT\}\}': '고객 지원팀 연락처',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text

# 번역 함수
def translate_to_korean(text):
    try:
        return translator.translate(text, dest='ko').text
    except Exception as e:
        st.error(f"번역 중 오류 발생: {e}")
        return text

# 데이터셋 전처리 함수
def preprocess_function(examples, tokenizer):
    inputs = [f"질문: {q}" for q in examples["instruction"]]
    targets = [f"답변: {r}" for r in examples["response"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Fine-tuning 함수
@st.cache_resource
def fine_tune_model(_df, tokenizer, model):
    train_dataset = _df.to_dict(orient="list")
    train_dataset = preprocess_function(train_dataset, tokenizer)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    return model

# RAG 함수
def rag(query, df, embedding_model, seq2seq_model, tokenizer):
    query_embedding = embedding_model.encode([query])
    df['embedding'] = df['instruction'].apply(lambda x: embedding_model.encode(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], query_embedding)[0][0])
    most_similar = df.nlargest(1, 'similarity')
    
    context = most_similar['instruction'].values[0]
    input_text = f"질문: {context}\n답변:"
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = seq2seq_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer = clean_text(answer)
    answer = translate_to_korean(answer)
    return answer

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

df = load_data()
embedding_model = load_embedding_model()
tokenizer, seq2seq_model = load_seq2seq_model()

if tokenizer is None or seq2seq_model is None:
    st.error("모델 로딩에 실패했습니다. 앱을 다시 시작해주세요.")
else:
    if 'model_fine_tuned' not in st.session_state:
        with st.spinner('모델을 Fine-tuning 중입니다. 잠시만 기다려주세요...'):
            seq2seq_model = fine_tune_model(df, tokenizer, seq2seq_model)
        st.session_state.model_fine_tuned = True
        st.success('Fine-tuning이 완료되었습니다!')

    query = st.text_input("질문을 입력하세요:")

    if query:
        answer = rag(query, df, embedding_model, seq2seq_model, tokenizer)
        st.write("답변:", answer)
