import google.generativeai as genai

genai.configure(api_key="API_KEY")

gen = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 500,
    "max_output_tokens": 10,
}

model = genai.GenerativeModel('gemini-pro')
encryption='SprYr8+teAKJJKTOqI44UOErWG/va9FtzUTEuTtMr4G1/Fb+vG3vuOYtpoO+irUf4y+x0iG5qhqMH/Zucag2yw=='

prompt = f"decrypt the following encryption and give the complete decrypted text. {encryption}"
response = model.generate_content(prompt,generation_config=gen)
op=response.text
print(op)
