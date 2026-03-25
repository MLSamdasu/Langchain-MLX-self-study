# 답변과 함께 어떤 문서에서 정보를 가져왔는지 출처를 반환 ㅓ
"""
기존 체인을 수정하지 않고, 바깥에서 감싸서 기능을 추가함
여기서 중요한건 UX관점 추가임. llm에는 출처가 다시 들어가지 않음
병렬로 처리되어서 UX 향상일 뿐임
출처 정보는 이미 context에 포함되어 들어감
"""

rag_with_sources = RunnableParallel(
    answer=rag_chain,  # 기존 RAG
    sources=retriever,  # 원본 문서
)

# 메타데이터에서 출처 추출
# result는 딕셔너리로 answer, sources 에 대한 정보가 있음
result = rag_with_sources.invoke("LCEL이란 무엇인가?")

# 답변 출력
print("== 답변 ==")
print(result["answer"])

# 출처 출력
print("\n== 출처 ==")
for doc in rasult["sources"]:
    src = doc.metadata.get("source", "unknown")  # 파일 경로
    page = doc.metadata.get("page", "?")  # 페이지 번호
    print(f" - {src} (p.{page})")
