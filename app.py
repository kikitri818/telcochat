import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv("hf://datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/bitext-telco-llm-chatbot-training-dataset.csv")
    return df.head(1000)  # 데이터 크기 제한

# 모델 로드 함수
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Fine-tuning 함수
def fine_tune_model(model, train_data):
    train_examples = [InputExample(texts=[row['instruction'], row['response']]) for _, row in train_data.iterrows()]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    return model

# Streamlit 앱
st.title("통신사 고객센터 챗봇")

# 데이터 및 모델 로드
df = load_data()
model = load_model()

# Fine-tuning
if 'model_fine_tuned' not in st.session_state:
    st.session_state.model_fine_tuned = False

if not st.session_state.model_fine_tuned:
    with st.spinner('모델을 Fine-tuning 중입니다...'):
        model = fine_tune_model(model, df)
    st.session_state.model_fine_tuned = True
    st.success('Fine-tuning이 완료되었습니다!')

# 사용자 입력 및 응답 생성
query = st.text_input("질문을 입력하세요:")

if query:
    # 가장 유사한 질문 찾기
    df['embedding'] = df['instruction'].apply(lambda x: model.encode(x))
    query_embedding = model.encode(query)
    df['similarity'] = df['embedding'].apply(lambda x: model.util.cos_sim(x, query_embedding)[0][0].item())
    most_similar = df.nlargest(1, 'similarity')
    
    # 응답 생성
    answer = most_similar['response'].values[0]
    st.write("답변:", answer)
