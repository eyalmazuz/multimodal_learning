import json
import requests

with open('<WEBHOOK/PATH/', 'w') as f:
    webhook_url = f.read().strip()

def send_message(text: str):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'content': text})
    requests.post(url=webhook_url, data=payload, headers=headers)
