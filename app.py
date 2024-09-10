import streamlit as st
import requests
from bs4 import BeautifulSoup

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
    for topic in data.get('RelatedTopics', []):
        if isinstance(topic, dict) and 'Text' in topic:
            results.append(topic['Text'])
    return results[:3]  # 최대 3개의 결과만 반환

def generate_response(query, tworld_results, web_results):
    response = f"'{query}'에 대한 검색 결과입니다:\n\n"
    
    if tworld_results:
        response += "T world 웹사이트 검색 결과:\n"
        for i, result in enumerate(tworld_results, 1):
            response += f"{i}. {result}\n"
        response += "\n"
    
    if web_results:
        response += "웹 검색 결과:\n"
        for i, result in enumerate(web_results, 1):
            response += f"{i}. {result}\n"
    
    if not tworld_results and not web_results:
        response = f"죄송합니다. '{query}'에 대한 정보를 찾지 못했습니다. 다른 방식으로 질문을 해보시거나, 고객센터에 문의해 주세요."
    
    return response

def main():
    st.title("텔코 챗봇")

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        tworld_results = search_tworld(user_input)
        web_results = web_search(user_input)
        response = generate_response(user_input, tworld_results, web_results)
        
        st.write("챗봇 응답:")
        st.write(response)
        if tworld_results:
            st.write("위 답변은 T world 웹사이트와 웹 검색을 참고했습니다.")
        elif web_results:
            st.write("위 답변은 웹 검색을 참고했습니다.")
        else:
            st.write("검색 결과가 없습니다.")

if __name__ == "__main__":
    main()
