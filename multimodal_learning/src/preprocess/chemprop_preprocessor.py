import json
import os
from typing import List

import numpy as np
import pandas as pd
from pandas import HDFStore
from tqdm import tqdm

from src.utils.utils import read_h5


def create_prediction_splits(df,target_att_name,set_name="set",num_test_sets=10,random_state=None,include_pos=False):
    assert set not in df.columns,'The column for target already exists in the dataframe'
    if not include_pos:
        df_test = df[df[target_att_name] != 1]
    else:
        df_test = df
    assert len(
        df_test) >= num_test_sets, 'The required number of test sets is larger than the number of negative cases'
    df_test = df_test.sample(frac=1, random_state=random_state)  # shuffle the dataframe
    df_test['index_row'] = np.arange(1, len(df_test) + 1)
    df_test[set_name] = df_test['index_row'].apply(lambda x: x % num_test_sets)
    df_test = df_test.drop(['index_row'], axis=1)


    df_train = df[df[target_att_name] == 1]
    df_train.loc[:,set_name] = 'train'
    if include_pos:
        ans = df_test
    else:
        ans = pd.concat([df_train,df_test])
    return ans

def preprocess_data(data_path: str, modalities_path: str, save_path: str, version: str) -> None:

    # main_target = 'cancer_clinical_trials'#cancer_clinical_trials
    # file = 'labels'

    main_outcome = pd.read_csv(f'{data_path}/labels.csv', index_col=0)
    # main_outcome = pd.read_csv(os.path.join('output','targets',main_target,'raw',file+'.csv'),index_col=0)

    if len(main_outcome.columns)>1:
        print('only one target attt is supported!!!!! Removing cols at end')
        main_outcome = main_outcome[[main_outcome.columns[0]]]


    main_outcome = main_outcome[~main_outcome.index.duplicated(keep='first')]
    print('removing',main_outcome.iloc[:,0].isna().sum(),'drugs with unknown class')
    main_outcome = main_outcome[~main_outcome.iloc[:,0].isna()]



    store = HDFStore(os.path.join(modalities_path, f'modalities_dict_{version}.h5'))
    print(store.info())
    X_unlabled = store['df']
    X_unlabled = X_unlabled.loc[~X_unlabled.index.duplicated(keep='first')]  # four drugs appear twice....
    modalities_df = store['modalities']
    store.close()
    data = X_unlabled[['Smiles']]

    smiles = data[~data.Smiles.isna()].drop_duplicates(subset='Smiles')
    ans = smiles[['Smiles']].join(main_outcome,how='inner')

    ans.to_csv(os.path.join(save_path, 'labels_training_set'+'.csv'),index=False)
    ans.to_csv(os.path.join(save_path, 'labels_training_set_w_drugbank_id'+'.csv'),index=True)



    current_feature_df = create_prediction_splits(ans, main_outcome.columns[0],num_test_sets=3)
    assert 'train' in current_feature_df.set.values
    for current_infer_set in set(list(x for x in current_feature_df.set.values if x != 'train')):
        current_set_df = current_feature_df[current_feature_df.set != current_infer_set]
        current_set_df.drop('set',axis=1).to_csv(os.path.join(save_path, str(current_infer_set) + '_train_labels.csv'),index=False)

        current_set_df = current_feature_df[current_feature_df.set == current_infer_set]
        current_set_df.drop('set',axis=1).to_csv(os.path.join(save_path, str(current_infer_set) + '_infer_labels.csv'))


    smiles.loc[~smiles.index.isin(ans.index)].drop_duplicates(subset='Smiles').to_csv(os.path.join(save_path, 'labels_infer_drugbank'+'.csv'),index=True)


def create_data_features(feature_df: pd.DataFrame,
                         feature_name: str,
                         data_path: str,
                         modalities_path: str,
                         files: List[str],
                         similarity_dict_path: str,
                         remove_unapproved_drugs: bool=True):

    h5 = read_h5(modalities_path)
    df = h5['/df']
    h5.close()

    approved_drugs = df[df['Group: approved'] == True].index.tolist()
    feature_df = feature_df[feature_df.index.isin(approved_drugs)]

    print(f'creating feature: {feature_name}')
    for file_ in files:
        feature_save_path = f'{data_path}/{file_}_{feature_name}.csv'
        if not os.path.exists(feature_save_path):
            path = os.path.join(data_path, f'{file_}.csv')
            df = pd.read_csv(path, index_col=0)
            # features_df = pd.read_csv(feature_path)
            with open(similarity_dict_path, 'r') as f:
                similarity_dict = json.load(f)
            
            # TODO remove un approved drugs from feature_df
            extra_df = pd.DataFrame()
            for drug_id in tqdm(df.index):
                found_feature = False
                if drug_id in feature_df.index:
                    drug_feature = feature_df[feature_df.index == drug_id]
                    extra_df = extra_df.append(drug_feature)
                    found_feature = True
                else:
                    if drug_id in similarity_dict and similarity_dict[drug_id]:
                        similar_drugs = similarity_dict[drug_id]
                        for (similar_drug, _) in similar_drugs:
                            if similar_drug in feature_df.index:
                                drug_feature = feature_df[feature_df.index == similar_drug]
                                drug_feature.name = drug_id
                                extra_df = extra_df.append(drug_feature)
                                found_feature = True
                                # print(f'found similar drug to {drug_id}: {similar_drug}')
                                break

                if not found_feature:
                    # print(f'No feature for {drug_id} and no similars in features')
                    mean_feature = feature_df.mean(axis=0)
                    mean_feature.name = drug_id
                    extra_df = extra_df.append(mean_feature)
            
            extra_df.to_csv(feature_save_path, index=False)

        

