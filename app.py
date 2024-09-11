import streamlit as st
import pandas as pd
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    dataset = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset", split="train")
    df = pd.DataFrame(dataset)
    if len(df) > 1000:  # 데이터가 1000개 이상이면 샘플링
        return df.sample(n=1000, random_state=42)
    return df

df = load_data()

# Sentence Transformer 모델 로드
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

sentence_model = load_sentence_transformer()

# 임베딩 생성
@st.cache_data
def create_embeddings(_df, _model):
    return _model.encode(_df['instruction'].tolist())

embeddings = create_embeddings(df, sentence_model)

# KoGPT 모델 및 토크나이저 로드
@st.cache_resource
def load_kogpt_model():
    model_name = "skt/kogpt2-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_kogpt_model()

# Fine-tuning
@st.cache_resource
def fine_tune_model(_df, _tokenizer, _model):
    dataset = Dataset.from_pandas(_df)
    
    def tokenize_function(examples):
        prompts = [f"질문: {q}\n답변: {r}" for q, r in zip(examples["instruction"], examples["response"])]
        return _tokenizer(prompts, truncation=True, padding=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=_tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    return _model

model = fine_tune_model(df, tokenizer, model)

# RAG 함수
def rag(query, top_k=3):
    try:
        query_embedding = sentence_model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        context = "\n".join(df.iloc[top_indices]['response'].tolist())
        
        prompt = f"""다음은 고객 질문에 대한 관련 정보입니다:

{context}

고객 질문: {query}

위 정보를 바탕으로 고객 질문에 대해 간결하고 정확하게 답변해주세요. 불필요한 정보는 제외하고 핵심만 말씀해 주세요:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.95)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        answer = response.split("핵심만 말씀해 주세요:")[-1].strip()
        
        return answer
    except Exception as e:
        return f"죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다: {str(e)}"

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

user_input = st.text_input("질문을 입력하세요:")

if user_input:
    answer = rag(user_input)
    st.write("답변:", answer)
    
