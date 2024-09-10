import streamlit as st
import requests

# Google Custom Search API 키와 검색 엔진 ID를 설정합니다.
# 실제 사용 시에는 이 값들을 환경 변수로 관리하는 것이 좋습니다.
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
GOOGLE_CSE_ID = "YOUR_GOOGLE_CSE_ID"

def web_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={query}"
    response = requests.get(url)
    data = response.json()
    
    results = []
    if 'items' in data:
        for item in data['items'][:3]:  # 상위 3개 결과만 사용
            results.append(item['snippet'])
    
    return results

def generate_response(query, search_results):
    if search_results:
        response = f"'{query}'에 대한 검색 결과입니다:\n\n"
        for i, result in enumerate(search_results, 1):
            response += f"{i}. {result}\n\n"
        return response
    else:
        return f"죄송합니다. '{query}'에 대한 정보를 찾지 못했습니다. 다른 방식으로 질문을 해보시거나, 고객센터에 문의해 주세요."

def main():
    st.title("텔코 챗봇")

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        search_results = web_search(user_input)
        response = generate_response(user_input, search_results)
        
        st.write("챗봇 응답:")
        st.write(response)
        if search_results:
            st.write("위 답변은 웹 검색을 참고했습니다.")
        else:
            st.write("검색 결과가 없습니다.")

if __name__ == "__main__":
    main()
