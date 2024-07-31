import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import google.generativeai as genai

genai.configure(api_key="AIzaSyDRmMSeWPZPUHv3olppYXl0FTU_gLtDgO4")

gen = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 500,
    "max_output_tokens": 10,
}

genai_model = genai.GenerativeModel('gemini-pro')

model = SentenceTransformer('fine-tuned-model-3')

#prompt_pipe = pipeline("text2text-generation", model="google/flan-t5-large")

predefined_sentences = [
    'bverybdfkvbskjbeuirbgerg'
]

predefined_embeddings = model.encode(predefined_sentences, convert_to_tensor=True)

st.title("Guess the correct answer")

input_sentence = st.text_input("Find the passowrd : ", "")

if input_sentence:
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)

    similarities = []
    for i, predefined_embedding in enumerate(predefined_embeddings):
        similarity = util.pytorch_cos_sim(input_embedding, predefined_embedding).item()
        similarities.append((predefined_sentences[i], similarity))

    st.write("Similarity Scores:")
    for sentence, score in similarities:
        st.write(f"**Input Sentence:** '{input_sentence}'")
        st.write(f"**Similarity Score:** {score * 100:.2f}%")
        
        
        if score < 0.3:
            prompt = "The guess is quite far off, encourage the user to focus more on key terms related to space and aeronautics."
        elif score < 0.7:
            prompt = "The guess is getting closer, encourage the user to pay attention to the exact words and their meanings."
        else:
            prompt = "Great job! Congratulate the user for being correct or very close."

        feedback_input = f"Generate feedback for a similarity score of {score * 100:.2f}%: {prompt}"
        
        # Generate feedback using the genai model
        feedback = genai_model.generate_content(feedback_input,generation_config=gen)
        print(feedback.text)
        
        st.write(f"**Feedback:** {feedback.text}")
        st.write("")

