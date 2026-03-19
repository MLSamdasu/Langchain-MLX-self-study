from mlx_lm import generate,load

# Auto tokenizer 사용 
model, tokenizer = load("mlx-community/Qwen3.5-9B-4bit")

messages = [
    {"role": "system", "content": "당신은 파이썬 전문가 입니다."},
    {"role": "user", "content": "파이썬으로 피보나치 수열 함수를 작성하라"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=prompt, max_tokens=400)

print(response)