import streamlit as st
import requests
from bs4 import BeautifulSoup

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

def generate_response(query, search_results):
    if search_results:
        response = f"'{query}'에 대한 검색 결과입니다:\n\n"
        for i, result in enumerate(search_results, 1):
            response += f"{i}. {result}\n\n"
        return response
    return f"죄송합니다. '{query}'에 대한 정보를 찾지 못했습니다. 다른 방식으로 질문을 해보시거나, 고객센터에 문의해 주세요."

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
