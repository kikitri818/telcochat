import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from deep_translator import GoogleTranslator
import requests
from bs4 import BeautifulSoup

# ... (이전 함수들은 그대로 유지) ...

def search_tworld(query):
    url = "https://www.tworld.co.kr/web/home"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    results = []
    keywords = query.lower().split()
    for elem in soup.find_all(['p', 'li', 'h3', 'h4', 'div', 'span']):
        text = elem.get_text().strip().lower()
        if any(keyword in text for keyword in keywords):
            results.append(elem.get_text().strip())
    
    return results[:3]  # 최대 3개의 결과만 반환

def web_search(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    data = response.json()
    
    results = []
    if data.get('Abstract'):
        results.append(data['Abstract'])
    if data.get('RelatedTopics'):
        for topic in data['RelatedTopics'][:2]:
            if 'Text' in topic:
                results.append(topic['Text'])
    
    return results

def perform_search(query, df, sentence_transformer, translator):
    # 1. Fine-tuning 데이터 검색
    relevant_context = retrieve_relevant_context(query, df, sentence_transformer)
    if not relevant_context.empty and relevant_context.iloc[0]['similarity'] > 0.7:
        response = relevant_context.iloc[0]['response']
        return translate_text(response, translator), "Fine-tuning 데이터"
    
    # 2. T world 웹사이트 검색
    tworld_results = search_tworld(query)
    if tworld_results:
        response = "\n".join(tworld_results)
        return response, "T world 웹사이트"
    
    # 3. 일반 웹 검색
    web_results = web_search(query)
    if web_results:
        response = "\n".join(web_results)
        return translate_text(response, translator), "웹 검색"
    
    return "죄송합니다. 해당 질문에 대한 정확한 답변을 찾지 못했습니다.", "검색 결과 없음"

# ... (main 함수는 그대로 유지) ...
