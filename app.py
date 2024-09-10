import streamlit as st
import requests
from bs4 import BeautifulSoup

def web_search(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    results = []
    for g in soup.find_all('div', class_='g'):
        snippet = g.find('div', class_='IsZvec')
        if snippet:
            results.append(snippet.text)
    
    return results[:3]  # 최대 3개의 결과만 반환

def generate_response(query, search_results):
    if search_results:
        response = f"'{query}'에 대한 검색 결과입니다:\n\n"
        for i, result in enumerate(search_results, 1):
            response += f"{i}. {result}\n\n"
        return response
    return f"죄송합니다. '{query}'에 대한 정보를 찾지 못했습니다."

def main():
    st.title("텔코 챗봇")

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        search_results = web_search(user_input)
        response = generate_response(user_input, search_results)
        
        st.write("챗봇 응답:")
        st.write(response)
        st.write("위 답변은 웹 검색을 참고했습니다.")

if __name__ == "__main__":
    main()
