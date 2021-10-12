import json
import pandas as pd
import requests

webhook = None

def send_message(text: str):
    global webhook
    if webhook is None:
        with open('./data/txt/webhook.txt', 'w') as f:
            webhook_url = f.read().strip()
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'content': text})
    requests.post(url=webhook_url, data=payload, headers=headers)

def read_h5(path: str='./data/DTI/h5/modalities_dict_5.1.8.h5'):
    # Read data. Created using create_data_files.py
    store = pd.HDFStore(path)
    print(store.info())
    print(store.keys())
    return store