import requests

def get_ai_response():
    url = 'http://127.0.0.1:8080/v1/chat/completions'

    payload = {
            "model": "gpt-3.5-turbo", 
            "messages": [
        {
            "role": "system", 
            "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
        },
        {
            "role": "user",
            "content": "Write a limerick about python exceptions"
        }]}

    r = requests.post(url, json=payload)

    message = r.json()['choices'][0]['message']['content']

    return message
