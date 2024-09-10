import streamlit as st
import requests
from bs4 import BeautifulSoup

def web_search(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    data = response.json()
    results = []
    if 'Abstract' in data and data['Abstract']:
        results.append(data['Abstract'])
    if 'RelatedTopics' in data:
        for topic in data['RelatedTopics'][:2]:  # 최대 2개의 관련 토픽 추가
            if 'Text' in topic:
                results.append(topic['Text'])
    return results if results else [f"'{query}'에 대한 정보를 찾지 못했습니다."]

def search_tworld(query):
    url = 'https://www.tworld.co.kr/web/home'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    results = []
    keywords = query.lower().split()
    for elem in soup.find_all(['p', 'li', 'h3', 'h4', 'div', 'span']):
        text = elem.get_text().strip().lower()
        if any(keyword in text for keyword in keywords):
            results.append(elem.get_text().strip())
    return results[:3]  # 최대 3개의 결과만 반환

def perform_search(query):
    # 1. T world 웹사이트 검색
    tworld_results = search_tworld(query)
    if tworld_results:
        return "\n".join(tworld_results), "T world 웹사이트"
    
    # 2. 일반 웹 검색
    web_results = web_search(query)
    if web_results:
        return "\n".join(web_results), "웹 검색"
    
    return "죄송합니다. 해당 질문에 대한 정확한 답변을 찾지 못했습니다.", "검색 결과 없음"

def main():
    st.title("텔코 챗봇")

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        response, source = perform_search(user_input)
        
        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}를 참고했습니다.")

if __name__ == "__main__":
    main()
