import requests

url = 'http://127.0.0.1:8080/completion'

#payload = '{"prompt": "Building a website can be done in 10 simple steps:","n_predict": 128}'

payload = {"prompt": "User: helloooo how are you doing ai friend? Assistant:", "stop": "User:"}

#headers = {'Content-Type': 'application/json'}

r = requests.post(url, json=payload)

if (r.status_code == 200):
  print(r.json()['content'])
else:
  print("Error: ", r.status_code)