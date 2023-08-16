from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
from dotenv import find_dotenv, load_dotenv
import requests
from datetime import datetime
from playsound import playsound
import speech_recognition as sr

load_dotenv(find_dotenv())

openai_api_key = os.getenv('OPENAI_API_KEY')
voice_id = os.getenv('VOICE_ID')
eleven_labs_api_key = os.getenv('ELEVEN_LABS_API_KEY')

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
        AI: 
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

def voice_message(message):
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + voice_id

    data = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": eleven_labs_api_key
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200 and response.content:
        ts = datetime.timestamp(datetime.now())

        with open(f'audio/{ts}.mp3', 'wb') as f:
            f.write(response.content)
        playsound(f'audio/{ts}.mp3')
        return response.content

# Web GUI
# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/send_message', methods=['POST'])
# def send_message():
#     human_input=request.form['human_input']
#     message = ai_response(human_input)
#     voice_message(message)
#     return message

# if __name__ == '__main__':
#     app.run(debug=True)

# obtain audio from the microphone
r = sr.Recognizer()

while True:
    try:
        with sr.Microphone() as mic:
            print("Listening...")
            r.adjust_for_ambient_noise(mic, duration=2)
            audio = r.listen((mic))

            try:
                transcript = r.recognize_whisper_api(audio, api_key=openai_api_key)

                if transcript == '' or transcript == None:
                    print("Failed to recognize voice.")
                    continue
                else:
                    message = ai_response(transcript)
                    print("*************************************************")
                    print(f"Me:\n{transcript}")
                    print("-------------------------------------------------")
                    print(f"AI: {message}")
                    print("*************************************************")
                    voice_message(message)
                    # return message
            except sr.RequestError as e:
                print("Could not request results from Whisper API")
    except sr.UnknownValueError():
        r = sr.Recognizer()
        continue