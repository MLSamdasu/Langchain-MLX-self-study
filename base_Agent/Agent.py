from langchain.agents import create_react_agent

agent = create_react_agent(
    model=llm,  # MLX 로컬 모델
    tools=tools,  # Agent가 사용할 도구 목록
)

result = agent.invoke({"messages": [{"role": "user", "content": "2의 20승은?"}]})
