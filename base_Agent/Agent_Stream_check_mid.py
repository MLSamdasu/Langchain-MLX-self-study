"""
이 부분은 디버깅에 핵심임
Agent가 뭘 생각하고, 어떤 Tool을 선택했는지 실시간으로 볼 수 있음

event구조 이해
agent.stream()이 반환하는 event는 LangGraph의 node 실행 단위임
"agent"노드 : LLM이 실행되는 단계. tool_calls가 있으면 Tool 호출 결정, 없으면 최종 답변
"tools"노드 : 실제 Tool 함수가 실행되는 단계. ToolMessage가 반환됨

-> LangSmith 트레이싱과 직결되는 부분임
"""

# 실전 디버깅 패턴
# Agent가 무한루프에 빠질 때 확인하는 방법
# 9B 모델에서 Agent를 돌릴때 가장 흔한 문제가 Tool결과를 제대로 해석 못해서 같은 Tool을 반복 호출하는 것임
# max_iterations같은 안전장치 없으면 무한루프에 빠짐
step_count = 0
for event in agent.stream({"messages": [...]}):
    step_count += 1
    if step_count > 10:  # 비정상적으로 많은 스텝
        print("Agent가 루프에 빠진것으로 의심됨")
        break


# Agent 스트리밍 & 중간 단계
for event in agent.stream(
    {"messages": [{"role": "user", "content": "이 프로젝트의 README를 읽고 요약해줘."}]}
):
    for key, value in event.item():
        if key == "agent":
            msg = value["messages"][-1]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[TOOL] {tc['name']}({tc['args']})")
            else:
                print(f"[ANSWER] {msg.content}")
        elif key == "tools":
            msg = value["messages"][-1]
            print(f"[RESULT] {msg.content[:100]} . . . ")
