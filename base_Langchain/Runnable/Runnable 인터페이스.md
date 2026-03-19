### Runnable 인터페이스 
#### invoke()
- 하나 넣고 하나 받기 
- 다 만들고 한번에 줌 
```python
result = chain.invoke(["input" : "랭체인이 뭐야?"])
```

#### stream()
- 실시간 토큰 출력 
- invoke()와 다르게 ChatGPT처럼 글자가 타닥타닥 나옴 

```python
for chunk in chain.stream({"input": "Python의 장점 3가지"}):                        
      print(chunk, end="", flush=True)
```


#### batch()
- 여러 개 한꺼번에 병렬처리
```python
results = chain.batch([
      {"input": "Python 장점"},                                                       
      {"input": "Java 장점"},                                                         
      {"input": "Rust 장점"}
])   
```

#### ainvoke()
- 비동기 처리 
- 기다리는 동안 다른 일을 할 수 있게함 

```python
result = await chain.ainvoke({"input": "..."})  
```

> 이 메서드들이 체인 전체에도, 개별 부품에도 동일하게 쓸 수 있다는 게 중요:
```python
  chain.invoke(...)    # 체인 전체                      
  prompt.invoke(...)   # 프롬프트만                                                   
  llm.invoke(...)      # LLM만                          
  parser.invoke(...)   # 파서만         
```                                              
