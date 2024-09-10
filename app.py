import streamlit as st
import requests

def web_search(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    data = response.json()
    
    results = []
    if data['Abstract']:
        results.append(data['Abstract'])
    for topic in data['RelatedTopics']:
        if 'Text' in topic:
            results.append(topic['Text'])
        if len(results) == 3:
            break
    
    return results

def perform_search(query):
    results = web_search(query)
    if results:
        return "\n\n".join(results), "웹 검색"
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
