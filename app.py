import streamlit as st
import requests
import logging

logging.basicConfig(level=logging.INFO)

def web_search(query):
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if data.get('Abstract'):
            results.append(data['Abstract'])
        if data.get('RelatedTopics'):
            for topic in data['RelatedTopics'][:2]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append(topic['Text'])
        
        if not results:
            results.append(f"'{query}'에 대한 검색 결과를 찾을 수 없습니다.")
        
        logging.info(f"Web search results: {results}")
        return results
    except Exception as e:
        logging.error(f"Web search error: {str(e)}")
        return [f"검색 중 오류가 발생했습니다: {str(e)}"]

def perform_search(query):
    logging.info(f"Received query: {query}")
    results = web_search(query)
    response = "\n\n".join(results)
    logging.info(f"Final response: {response}")
    return response, "웹 검색"

def main():
    st.title("텔코 챗봇")

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        logging.info(f"User input: {user_input}")
        response, source = perform_search(user_input)
        
        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}를 참고했습니다.")

if __name__ == "__main__":
    main()
