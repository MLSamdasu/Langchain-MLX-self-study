# 필수 import
from langchain_openai import ChatOpenAI

# 옛날방식
# from langchain.agents import create_react_agent
from langchain.agents import create_agent

# 내 폴더에서 만든 파일명으로 import
from Tools_Decorator import calculate, search_web
from StructuredTool_Pydantic import file_reader

# MLX 로컬 모델
llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="X",
    model="mlx-community/Qwen3.5-9B-4bit",
    temperature=0,
)

# Tool 목록
tools = [calculate, search_web, file_reader]

# Agent 생성
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="도구 결과를 반드시 자연어로 요약하라. 빈 응답 금지.",
)

# 실행
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "2+2는 얼마인가요?",
            }
        ]
    }
)

print(result["messages"][-1].content)
