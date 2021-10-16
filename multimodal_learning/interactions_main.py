import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras.layers import embeddings
import sys
print(sys.path)
import random 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.features.drug_interactions.get_features import get_features
from src.persistant.readers.xml_reader.drugbank_reader import DrugReader
from src.preprocess.xml_preprocessor import DrugPreprocessor
from src.datasets.drug_interactions.dataset_builder import get_dataset, DatasetTypes
from src.training.train import Trainer
from src.training.evaluate import predict, predict_tta
from src.models.drug_interactions.model_builder import get_model
from src.utils.utils import send_message

# tf.random.set_seed(0)
# np.random.seed(0)
# random.seed(0)

def main():

    path = './data/DrugBankReleases'
    old_version = '5.1.6'
    new_version = '5.1.8'

    reader = DrugReader(path)

    old_drug_bank, new_drug_bank = reader.get_drug_data(old_version, new_version)
    preprocessor =  DrugPreprocessor()

    old_drug_bank = preprocessor.get_preprocessed_drug_bank(old_drug_bank, f'{path}/{old_version}')
    new_drug_bank = preprocessor.get_preprocessed_drug_bank(new_drug_bank, f'{path}/{new_version}')

    print(f'Old drug bank num of drugs: {len(old_drug_bank.drugs)}')
    print(f'New drug bank num of drugs: {len(new_drug_bank.drugs)}')

    print(f'Num of old drug bank interactions: {sum(map(len, [drug.interactions for drug in old_drug_bank.drugs]))}')
    print(f'Num of old drug bank interactions: {sum(map(len, [drug.interactions for drug in new_drug_bank.drugs]))}')

    smiles_length = [len(drug.smiles) for drug in new_drug_bank.drugs if drug.smiles]

    print(f'max drug smile length: {max(smiles_length)}')

    types = {
        '1': DatasetTypes.AFMP,
        '2': DatasetTypes.ONEHOT_SMILES,
        '3': DatasetTypes.CHAR_2_VEC,
    }
    dataset_type = types[os.environ['SLURM_ARRAY_TASK_ID']]
    dataset_type_str = str(dataset_type).split(".")[1]
    feature_config = {
        # CNN features
        "atom_size": 300,
        "atom_info": 21,
        "struct_info": 21,
        # Char2Vec feature
        "embedding_size": 256,
        "window": 5,
        "min_count": 1,
        "workers": 8,
        "epochs": 5,
    }

    features = get_features(dataset_type, **feature_config)
    send_message(f'Starting {str(dataset_type)}')
    send_message(f'Starting {dataset_type_str}')
    print(f'Starting {dataset_type_str}')

    (train_dataset, validation_dataset,
        test_new_old_similar_dataset,
        test_new_new_similar_dataset,
        # test_all_similar_dataset,
        metadata) = get_dataset(old_drug_bank,
                                new_drug_bank,
                                feature_list=features,
                                sample=True,
                                epoch_sample=False,
                                neg_pos_ratio=1.0,
                                validation_size=0.2,
                                batch_size=1024,
                                atom_size=300,
                                data_path='./data/DDI/csvs/',)

    metadata['embedding_size'] = 2
    model = get_model(dataset_type, **metadata)

    trainer = Trainer(epoch_sample=False, balance=False)

    trainer.train(model, train_dataset, validation_dataset, epochs=3, dataset_type=dataset_type_str)

    # predict_tta(model, test_new_old_similar_dataset, dataset_type=f'L{dataset_type_str}_2', save_path='./data/DDI/csvs/results/All_Data/TTANewOld', save=True)

    #predict_tta(model, test_new_new_similar_dataset, dataset_type=f'L{dataset_type_str}_2', save_path='./data/DDI/csvs/results/All_Data/TTANewNew', save=True)
    
    # predict(model, test_new_old_similar_dataset, dataset_type=dataset_type_str, save_path='./data/DDI/csvs/results/All_Data/NewOldNLCS', save=True)

    # predict(model, test_new_new_similar_dataset, dataset_type=dataset_type_str, save_path='./data/DDI/csvs/results/All_Data/NewNewNLCS', save=True)

    with open('./data/DDI/jsons/drug_emb_order.json', 'w') as f:
        json.dump(train_dataset.features['EmbeddingFeature'], f)

    drug_emb = model.drug_embedding.get_weights()
    np.save('./data/DDI/npys/2d_embs.npy', drug_emb[0])

if __name__ == "__main__":
    main()
