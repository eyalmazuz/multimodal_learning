from datetime import datetime
import os
import shutil

import chemprop
import pandas as pd

from src.persistant.readers.db_reader.db_reader import get_chembl_data
from src.preprocess.chemprop_preprocessor import prepare_cancer_data, create_data_features, prepare_chembl_data
from src.utils.description_utils import enrich_predicitons
from src.features.drug_target.TargetPCAFeature import TargetPCAFeature
from src.features.drug_target.DDIFeature import DDIFeature
from src.utils.configs import get_task_config

def main():

    cur_date = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    TASK = 'target'

    drugbank_version = '5.1.8'
    modalities_path = './data/DTI/h5'
    cancer_path = './data/DTI/cancer_clinical_trials'
    features_path = './data/DTI/features'
    
    checkpoint_path = f'./data/DTI/target_prediction/cps'
    train_path = f'./data/DTI/target_prediction/targets.csv'
    test_path = f'./data/DTI/target_prediction/test.csv'

    preds_save_path = f'./data/DTI/target_prediction/predictions/{cur_date}_predictions.csv'

    ensemble_size = '2'
    num_folds = '2'
    
    use_additional_features = False

    if TASK == "target": 
        df = get_chembl_data('./data/chembl_29_sqlite/chembl_29.db',
                        ['CHEMBL2189121', 'CHEMBL4096'],
                        ['IC50', 'Ki'],)
                        # './data/DTI/target_prediction/')

        if not os.path.exists(train_path):
            prepare_chembl_data(df, train_path)
    
    elif TASK == "cancer":
        prepare_cancer_data(version=drugbank_version,
                              modalities_path=modalities_path,
                              path=cancer_path)

    task_eval_arguments, task_train_arguments = get_task_config(TASK)

    features = [
        TargetPCAFeature,
        DDIFeature
        ]

    features_params = {
        # Target PCA params
        'pca_dim': 64,
        'modalities_path': f'{modalities_path}/modalities_dict_{drugbank_version}.h5',
        'features_path': features_path,
        
        # DDI feautre params
        'emb_dim': 256,
        'old_version': '5.1.8',
        'drugbank_path': './data/DrugBankReleases/',
        'sample': True,
        'epoch_sample': False,
        'neg_pos_ratio': 1.0,
        'validation_size': 0.2,
        'batch_size': 1024,
        'atom_size': 300,
        'DDI_data_path': './data/DDI/csvs'
    }
    additional_features = []

    if features and use_additional_features:
        for feat in features:
            feature = feat(**features_params)
            additional_features += [feature]
            features_df = feature()
            create_data_features(features_df, str(feature),
                                 f'{modalities_path}/modalities_dict_{drugbank_version}.h5',
                                [train_path,
                                test_path],
                                './data/DDI/jsons/similar_drugs_dict_all.json')
    
    features_names = ''            
    if use_additional_features:
        for i, feature in enumerate(additional_features):
            features_names += '_' + str(feature)

    eval_arguments = [
        '--data_path', f'{train_path}',
        '--num_workers', '8',
        '--split_sizes', '0.7', '0.1', '0.2',
        '--num_folds', f'{num_folds}',
        '--save_dir', f'{checkpoint_path}/{cur_date}_evaluation_{TASK}_{features_names}_checkpoints',
        '--smiles_column', 'Smiles',
        '--epochs', '30',
        '--ensemble_size', f'{ensemble_size}',
        '--save_preds',
        '--save_smiles_splits',
        '--config_path', f'./data/DTI/jsons/full_data_hyperparams_w_rkdit.json'
    ]

    eval_arguments += task_eval_arguments

    if features and use_additional_features:
        eval_arguments += ['--features_path']
        for feature in additional_features:
            last_dot_idx = train_path.rfind('.')
            file_path = train_path[:last_dot_idx]
            eval_arguments += [
                f'{file_path}_{str(feature)}.csv'
            ]
    else:
        eval_arguments += ['--features_generator', 'rdkit_2d_normalized',
                            '--no_features_scaling']

    eval_args = chemprop.args.TrainArgs().parse_args(eval_arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=eval_args, train_func=chemprop.train.run_training)

    train_arguments = [
        '--data_path', f'{train_path}',
        '--num_workers', '8',
        '--split_sizes', '0.9', '0.1', '0.0',
        '--save_dir', f'{checkpoint_path}/{cur_date}_full_data_{TASK}_{features_names}_checkpoints',
        '--smiles_column', 'Smiles',
        '--epochs', '30',
        '--ensemble_size', f'{ensemble_size}',
        '--save_preds',
        '--save_smiles_splits',
        '--config_path', f'./data/DTI/jsons/full_data_hyperparams_w_rkdit.json'
    ]

    train_arguments += task_train_arguments

    if features and use_additional_features:
        train_arguments += ['--features_path']
        for feature in additional_features:
            last_dot_idx = train_path.rfind('.')
            file_path = train_path[:last_dot_idx]
            train_arguments += [
                f'{file_path}_{str(feature)}.csv'
            ]

    else:
        train_arguments += ['--features_generator', 'rdkit_2d_normalized',
                            '--no_features_scaling']

    train_args = chemprop.args.TrainArgs().parse_args(train_arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=train_args, train_func=chemprop.train.run_training)
    print(mean_score, std_score)

    predict_arguments = [
        '--test_path', test_path,
        '--checkpoint_dir', f'{checkpoint_path}/{cur_date}_full_data_{TASK}_{features_names}_checkpoints',
        '--preds_path', f'{preds_save_path}',
        '--smiles_column', 'Smiles',
    ]

    if features and use_additional_features:
        predict_arguments += ['--features_path']
        for feature in additional_features:
            last_dot_idx = test_path.rfind('.')
            file_path = test_path[:last_dot_idx]
            predict_arguments += [
                f'{file_path}_{str(feature)}.csv'
            ]
    
    else:
        predict_arguments += ['--features_generator', 'rdkit_2d_normalized',
                            '--no_features_scaling']

    predict_args = chemprop.args.PredictArgs().parse_args(predict_arguments)
    preds = chemprop.train.make_predictions(args=predict_args)
    
    if use_additional_features:
        enrich_predicitons(preds_save_path, features_names)

if __name__ == "__main__":
    main()