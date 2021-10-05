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
    X_unlabled = store['/df']
    X_unlabled = X_unlabled.loc[~X_unlabled.index.duplicated(keep='first')]  # four drugs appear twice....
    modalities_df = store['/modalities']
    store.close()
    print('done reading from disk')
    return X_unlabled.loc[:,modalities_df.loc[modalities_df.modality.isin(['Target','Associated_condition','Smiles']),'feature']],modalities_df #reading only targets
