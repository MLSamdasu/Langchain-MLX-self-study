from mlx_lm import generate, load

model, tokenizer = load("mlx-community/Qwen3.5-9B-4bit")

prompt = "파이썬으로 피보나치 수열 함수를 작성하라"

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=200,
)
print(response)
