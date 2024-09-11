import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import torch
import faiss

@st.cache_resource
def load_data():
    df = pd.read_csv("https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/raw/main/bitext-telco-llm-chatbot-training-dataset.csv")
    return df

@st.cache_resource
def prepare_data_and_index(df):
    # Sentence transformer for encoding
    encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    # Encode instructions
    instruction_embeddings = encoder.encode(df['instruction'].tolist())
    
    # Create FAISS index
    index = faiss.IndexFlatL2(instruction_embeddings.shape[1])
    index.add(instruction_embeddings)
    
    return df, index, encoder

@st.cache_resource
def train_model(df):
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preprocess_function(examples):
        inputs = ["question: " + q for q in examples['instruction']]
        targets = examples['response']
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = df.to_dict(orient='list')
    dataset = preprocess_function(dataset)

    train_dataset = dataset
    
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
        train_dataset=train_dataset,
    )

    trainer.train()
    return model, tokenizer

st.title("텔코 고객센터 챗봇")

df = load_data()
df, index, encoder = prepare_data_and_index(df)
model, tokenizer = train_model(df)

user_input = st.text_input("질문을 입력하세요:")

if user_input:
    # RAG: Retrieve similar questions
    query_embedding = encoder.encode([user_input])
    D, I = index.search(query_embedding, k=3)
    
    context = " ".join(df.iloc[I[0]]['response'].tolist())
    
    # Generate response
    input_text = f"question: {user_input} context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    outputs = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.write(f"챗봇 응답: {response}")
