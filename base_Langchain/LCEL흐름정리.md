## 2.8 전체 LCEL 흐름도

### LCEL 체인 전체 흐름 (ASCII)

```
[Input Dict]
     |
     v
[PromptTemplate]       -- 변수를 프롬프트에 대입
     |
     v
[LLM / ChatModel]     -- 프롬프트를 모델에 전달
     |
     v
[OutputParser]         -- 응답을 원하는 형식으로 변환
     |
     v
[RunnableLambda]       -- 후처리 (선택)
     |
     v
[Final Output]
```

### 확장 흐름 (병렬 + 분기 포함)

```
                    [사용자 입력]
                         |
                         v
                  +-------------+
                  |   Branch?   |  <-- RunnableBranch (조건 분기)
                  +-------------+
                   /     |     \
                  v      v      v
            [코드체인] [번역체인] [일반체인]
                  \      |      /
                   v     v     v
                  +-------------+
                  |  Parallel?  |  <-- RunnableParallel (병렬 처리)
                  +-------------+
                   /           \
                  v             v
           [요약 체인]      [키워드 체인]
                  \             /
                   v           v
           {"summary": "...", "keywords": "..."}
```
