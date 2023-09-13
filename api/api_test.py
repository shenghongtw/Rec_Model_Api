import requests
import json

data = {"user": "2"}
data = json.dumps(data)
url = 'http://127.0.0.1:8000/submit'
r = requests.post(url, data=data)
print(r.content)
