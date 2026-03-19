from langchain_openai import ChatOpenAI

# 로컬 MLX 서버에 연결
llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-need",  # 로컬이라 불필요함
    model="mlx-community/Qwen3.5-9B-4bit",
    temperature=0.7,
    max_tokens=2048,
)

# 간단 테스트
response = llm.invoke("Python의 장잠 3가지")
# content -> LangChain이 정해놓은 속성.
# ChatOpenAI는 항상 AIMessage 객체 반환 -> 이 클래스 안에 .content 속성 정의
# 순수 답변 텍스트만 받게됨
print(response.content)
