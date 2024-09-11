import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import torch

@st.cache_resource
def load_data():
    df = pd.read_csv("https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/raw/main/bitext-telco-llm-chatbot-training-dataset.csv")
    return df.sample(n=1000, random_state=42)  # 1000개의 샘플을 사용

@st.cache_resource
def prepare_data(df):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: {
        "question": x["input"],
        "context": x["response"],
        "answer": x["intent"]
    })
    return dataset.train_test_split(test_size=0.2, seed=42)

@st.cache_resource
def train_model(dataset):
    model_name = "distilbert-base-uncased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["question"], examples["context"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()
    return model, tokenizer

st.title("텔코 고객센터 챗봇")

df = load_data()
st.write(f"로드된 데이터 샘플 수: {len(df)}")

dataset = prepare_data(df)
model, tokenizer = train_model(dataset)

user_input = st.text_input("질문을 입력하세요:")

if user_input:
    context = "이 챗봇은 텔코 회사의 고객 서비스를 지원합니다."  # 실제 상황에 맞는 컨텍스트로 대체해야 합니다
    inputs = tokenizer(user_input, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
    
    st.write(f"답변: {answer}")
