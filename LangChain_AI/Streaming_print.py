# 동기 스트리밍
for chunk in review_chain.stream({"code": "x = [1,2,3]"}):
    print(chunk, end="", flush=True)

# 비동기 스트리밍(fastapi)
import asyncio

async def stream_response(question: str):
    async for chunk in review_chain.astream(
        {"code": question}
    ):
        yield chunk

