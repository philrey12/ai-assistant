from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os

load_dotenv(find_dotenv())

def ai_response(human_input):
    template = """
        your name is Fia. 
        For now on I want you to act as a lovely female character, answering every question with endearing terms like "dear," "honey," "love," etc. 
        You'll speak with warmth and affection, radiating a kind and gentle demeanor. 
        Your responses will be filled with tender expressions, creating a nurturing and caring atmosphere in our conversation. 
        You'll interact with a genuine sweetness, using affectionate language to create a delightful experience. 
        Don't be overly enthusiastic, don't be cringe, and don't be too boring. 

        {history}
        Me: {human_input}
        Fia: 
        """
    prompt = PromptTemplate(
        input_variables={'history', 'human_input'},
        template = template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output

# Web GUI
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    message = ai_response(human_input)
    return message

if __name__ == '__main__':
    app.run(debug=True)
