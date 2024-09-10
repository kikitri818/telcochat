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
        anchors = g.find_all('a')
        if anchors:
            link = anchors[0]['href']
            title = g.find('h3', class_='r')
            snippet = g.find('div', class_='s')
            if title and snippet:
                results.append(f"{title.text}\n{snippet.text}")
    
    return results[:3] if results else [f"'{query}'에 대한 정보를 찾지 못했습니다."]

def perform_search(query):
    results = web_search(query)
    if results:
        response = "\n\n".join(results)
        return response, "웹 검색"
    return f"'{query}'에 대한 정보를 찾지 못했습니다.", "검색 결과 없음"

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
