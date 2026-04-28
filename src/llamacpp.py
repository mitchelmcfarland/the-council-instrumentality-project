import requests

url = 'http://127.0.0.1:8080/v1/chat/completions'

history = {
        "model": "gabemgooly",
        "messages": [
        {
            "role": "system", 
            "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
        }
        ]
    }


def get_ai_response(content):
    
    history["messages"].append({"role": "user", "content": content})

    r = requests.post(url, json=history)

    message = r.json()['choices'][0]['message']['content']

    role = r.json()['choices'][0]['message']["role"]

    history["messages"].append({"role": role, "content": message})

    print(history["messages"])

    return message
