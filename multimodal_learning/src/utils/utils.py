import json
import pandas as pd
import requests

from src.utils.identifiers_resolver import APIBasedIdentifiersResolver, DrugIdentifiersResolver, \
    WikiDataIdsResolver

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
    return store

def convert_to_IC50(pIC50) -> float:
    if pIC50 == "Invalid SMILES":
        return pIC50
    else:
        if isinstance(pIC50 ,str):
            pIC50 = float(pIC50)

    ic50 = 10 ** (9 - pIC50)
    
    return ic50

class add_to_df_drugBank_id():

    def add_drugBank_id(self, df, fieldName='generic_drug_name'):
        #resolver = DrugIdentifiersResolver(WikiDataIdsResolver('qid_to_drugbank.json', 'qid_to_pubchem.json'), APIBasedIdentifiersResolver())
        db2_ans = []
        pc_ans = []
        db_ans = []
        id_converter = WikiDataIdsResolver()
        print('working on wikipedia')
        for i,d_name in enumerate(df[fieldName]):
            # drugbank_id, pubchem_id = resolver.resolve_array(d_name)
            # print(id_converter.get_cid_and_dbid_by_name('aspirin'))
            drugbank_id, pubchem_id = id_converter.get_ids_by_name(d_name)
            db2_ans.append(drugbank_id)
            pc_ans.append(pubchem_id)
            if i%100==0:
                print('done',i,'of',len(df[fieldName]))

        print('working on API')
        id_converter = APIBasedIdentifiersResolver()
        for i,d_name in enumerate(df[fieldName]):
            res2 = id_converter.get_drug_bank_code_by_name(str(d_name))
            db_ans.append(res2)
            if i%100==0:
                print('done',i,'of',len(df[fieldName]))

        df['drugbank_id'] = [db2_ans[i] if x is None else x for i,x in enumerate(db_ans)]
        df['pubchem_id'] = pc_ans
        return df

def drug_names_to_drugBank(drug_names, x):
    #collects drugbankIDs from names
    y_nih = pd.DataFrame(
        {'generic_drug_name': [x if '(' not in x else x.split(' (')[0] for x in drug_names],
         'generic_drug_name2': [None if '(' not in x else x.split(' (')[1].replace(')', '') for x in
                                drug_names]})
    adder = add_to_df_drugBank_id()
    y_nih = adder.add_drugBank_id(y_nih)
    y_nih = y_nih.rename({'drugbank_id': 'drugBank_id1'}, axis=1)
    y_nih = adder.add_drugBank_id(y_nih, fieldName='generic_drug_name2')
    y_nih['drugbank3_id'] = [x if x is not None else y_nih['drugbank_id'].values[i] for i, x in
                             enumerate(y_nih['drugBank_id1'].values)]
    y_nih = y_nih['drugbank3_id']
    y_nih = y_nih[y_nih != 'None'].drop_duplicates()
    y_nih = pd.DataFrame(index=x.index).join(pd.DataFrame({'y': [1 for i in range(len(y_nih))]}, index=y_nih.values),
                                             how='left').fillna(0)
    y_nih = y_nih.y
    return y_nih