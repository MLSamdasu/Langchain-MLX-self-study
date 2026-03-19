from langchain_core.prompts import ChatPromptTemplate

"""
ChatPromptTemplate은 system/user/assistant 등 역할별 메시지를 구성합니다.
채팅 모델(Chat Model)에 맞는 형식을 자동으로 만들어주며,
MLX 채팅 모델과 또같은 system/user 구조입니다.
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 {topic} 전문가 입니다. 한국어로 답하시오"),
        ("user", "{question}"),
    ]
)

# 변수 대입 
massages = chat_prompt.format_messages(
    role="파이썬",
    question = "데코레이터란 무엇인지 10글자로 말해라."
)

print(prompt)