import os
import pickle
from pathlib import Path
import random
import sqlite3
from typing import List


import pandas as pd

from src.persistant.readers.db_reader.get_drugBank_features_from_DB import GetDrugBankFeaturesFromDB
from src.persistant.readers.db_reader.get_dense_vectors_features_from_DB import GetDenseVectorsFeaturesFromDB
from src.persistant.readers.db_reader.table_names import *

random.seed(30)

def get_h5_data(version: str, save_path: str) -> None:

    t = GetDrugBankFeaturesFromDB()
    ans,ans_modalities = t.combine_features([category_table,
                                            ATC_table_1,ATC_table_2,ATC_table_3,ATC_table_4,ATC_table_5,
                                            # ATC_table_1_description,ATC_table_2_description,ATC_table_3_description,ATC_table_4_description,
                                            enzyme_table,carrier_table,transporter_table,associated_condition_table,group_table,type_table],
                            dense_table_name_list=[tax_table,smiles_table, mol_weight_table],version=version,add_counts_to_sparse=True)


    ans_targets = t.get_target_features(version, column='target', value_type= 'binary')
    print(ans_targets.head())

    ans_targets.columns = ['Target: ' + x for x in ans_targets.columns]
    ans = ans.join(ans_targets,how='left')
    ans_modalities['Target'] = list(ans_targets.columns)


    def encode_and_bind(original_dataframe, feature_to_encode, ans_modalities,m):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]],dummy_na=True,prefix_sep=': ')
        res = pd.concat([original_dataframe, dummies],axis=1)
        res = res.drop([feature_to_encode], axis=1)
        ans_modalities[m].remove(feature_to_encode)
        ans_modalities[m]+=list(dummies.columns)
        return res, ans_modalities

    for c in list(ans_modalities['Taxonomy']):
        print('one hot encoding features',c)
        ans,ans_modalities = encode_and_bind(ans,c,ans_modalities,'Taxonomy')

    #Write results
    ans_modalities = pd.DataFrame({'modality':[x for x in sorted(ans_modalities.keys()) for y in ans_modalities[x]],'feature':[y for x in sorted(ans_modalities.keys()) for y in ans_modalities[x]]})

    for c in ans_modalities['feature'].values:
        assert c in ans.columns, "canot find column in df "+c

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    store = pd.HDFStore(os.path.join(save_path,'modalities_dict_'+version+'.h5'))
    store['df'] = ans
    store['modalities'] = ans_modalities
    store.close()

def get_chembl_data(db_path: str, target_ids: List[str]=None, standard_types: List[str]=None):
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT m.chembl_id AS compound_chembl_id,
    s.canonical_smiles,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.pchembl_value,
    act.potential_duplicate,
    t.chembl_id                    AS target_chembl_id,
    t.pref_name                    AS target_name,
    t.organism                     AS target_organism,
    bo.label as bao_label
    FROM compound_structures s
    JOIN molecule_dictionary m ON s.molregno = m.molregno
    JOIN compound_records r ON m.molregno = r.molregno
    JOIN docs d ON r.doc_id = d.doc_id
    JOIN activities act ON r.record_id = act.record_id
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary t ON a.tid = t.tid
    JOIN bioassay_ontology bo ON a.bao_format = bo.bao_id
    """
    if target_ids:
        target_ids = [f"'{target}'" for target in target_ids]
        query += "AND t.chembl_id IN ({','.join(target_ids)})"

    if standard_types:
        standard_types = [f"'{type}'" for type in standard_types]

    query += f"""
        AND m.chembl_id IN
            (SELECT DISTINCT
                m1.chembl_id
            FROM molecule_dictionary m1
                JOIN molecule_hierarchy mh ON mh.molregno = m1.molregno
                JOIN molecule_dictionary m2 ON mh.parent_molregno = m2.molregno)
        AND act.standard_type IN ({','.join(standard_types)})
        AND act.standard_type IN ('IC50')
        AND act.standard_units = 'nM'
        AND act.potential_duplicate = 0
        AND t.organism = 'Homo sapiens'
        AND act.standard_relation = '=';
    """

    df = pd.read_sql(query, conn)
    conn.close()
    df.to_csv('./foo.csv', index=False)

    return df

if __name__ == "__main__":
    get_h5_data('5.1.8', './data/h5/')