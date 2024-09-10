import requests
from bs4 import BeautifulSoup
import streamlit as st

def google_search(query):
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
    
    return results[:3]  # 최대 3개의 결과만 반환

def main():
    st.title("텔코 챗봇")

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        search_results = google_search(user_input)
        
        if search_results:
            st.write("챗봇 응답:")
            for i, result in enumerate(search_results, 1):
                st.write(f"{i}. {result}\n")
            st.write("위 답변은 웹 검색을 참고했습니다.")
        else:
            st.write("죄송합니다. 해당 질문에 대한 정보를 찾지 못했습니다. 다른 방식으로 질문을 해보시거나, 고객센터에 문의해 주세요.")

if __name__ == "__main__":
    main()
