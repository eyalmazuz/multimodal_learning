import os
import shutil

import chemprop
import pandas as pd

from src.persistant.readers.db_reader.db_reader import get_h5_data
from src.preprocess.chemprop_preprocessor import preprocess_data, create_data_features
from src.utils.utils import read_h5
from src.utils.description_utils import enrich_data
from src.features.drug_target.TargetPCAFeature import TargetPCAFeature
from src.features.drug_target.DDIFeature import DDIFeature

def main():
    version = '5.1.8'
    modalities_path = './data/DTI/h5'
    features_path = './data/DTI/features'

    path = './data/DTI/cancer_clinical_trials'
    checkpoint_path = f'{path}/cps'
    data_save_path = f'{path}/processed'
    
    use_additional_features = True
    features = [
        TargetPCAFeature,
        DDIFeature
        ]
    features_params = {
        # Target PCA params
        'pca_dim': 64,
        'modalities_path': f'{modalities_path}/modalities_dict_{version}.h5',
        'features_path': features_path,
        
        # DDI feautre params
        'emb_dim': 256,
        'old_version': '5.1.8',
        #'new_version': './data/DrugBankReleases/5.1.8',
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
    
    data_name = 'labels'
    ensemble_size = '2'
    num_folds = '10'

    if not os.path.exists(f'{modalities_path}/modalities_dict_{version}.h5'):
        get_h5_data(version=version, save_path=modalities_path)

    if not os.path.exists(f'{path}/processed/'):
        preprocess_data(data_path=f'{path}/raw',
                        modalities_path='./data/DTI/h5/',
                        version=version,
                        save_path=f'{path}/processed')

    if use_additional_features:
        for feat in features:
            feature = feat(**features_params)
            additional_features += [feature]
            features_df = feature()
            create_data_features(features_df, str(feature), data_save_path,
                                 f'{modalities_path}/modalities_dict_{version}.h5',
                                    ['labels_training_set_w_drugbank_id', 'labels_infer_drugbank'],
                                    './data/jsons/similar_drugs_dict_all.json')
                

    if os.path.exists(f'{checkpoint_path}/evaluation_{data_name}_checkpoints'):
        shutil.rmtree(f'{checkpoint_path}/evaluation_{data_name}_checkpoints')
    
    eval_arguments = [
        '--data_path', f'{data_save_path}/{data_name}_training_set.csv',
        # '--features_generator', 'rdkit_2d_normalized',
        '--no_features_scaling',
        '--dataset_type', 'classification',
        '--num_workers', '8',
        '--extra_metrics', 'prc-auc',
        '--split_type', 'scaffold_balanced',
        '--split_sizes', '0.7', '0.1', '0.2',
        '--num_folds', f'{num_folds}',
        '--save_dir', f'{checkpoint_path}/evaluation_{data_name}_checkpoints',
        '--smiles_column', 'Smiles',
        '--epochs', '30',
        '--ensemble_size', f'{ensemble_size}',
        '--save_preds',
        '--save_smiles_splits',
        '--config_path', f'./data/DTI/jsons/full_data_hyperparams_w_rkdit.json'
    ]

    if use_additional_features:
        eval_arguments += ['--features_path']
        for feature in additional_features:
            eval_arguments += [
                f'{data_save_path}/{data_name}_training_set_w_drugbank_id_{str(feature)}.csv'
            ]

    eval_args = chemprop.args.TrainArgs().parse_args(eval_arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=eval_args, train_func=chemprop.train.run_training)

    if os.path.exists(f'{checkpoint_path}/full_data_{data_name}_checkpoints'):
        shutil.rmtree(f'{checkpoint_path}/full_data_{data_name}_checkpoints')
        
    train_arguments = [
        '--data_path', f'{data_save_path}/{data_name}_training_set.csv',
        # '--features_generator', 'rdkit_2d_normalized',
        '--no_features_scaling',
        '--dataset_type', 'classification',
        '--num_workers', '8',
        '--extra_metrics', 'prc-auc',
        '--split_type', 'scaffold_balanced',
        '--split_sizes', '0.9', '0.1', '0.0',
        '--save_dir', f'{checkpoint_path}/full_data_{data_name}_checkpoints',
        '--smiles_column', 'Smiles',
        '--epochs', '30',
        '--ensemble_size', f'{ensemble_size}',
        '--save_preds',
        '--save_smiles_splits',
        '--config_path', f'./data/DTI/jsons/full_data_hyperparams_w_rkdit.json'
    ]

    if use_additional_features:
        train_arguments += ['--features_path']
        for feature in additional_features:
            train_arguments += [
                f'{data_save_path}/{data_name}_training_set_w_drugbank_id_{str(feature)}.csv'
            ]

    train_args = chemprop.args.TrainArgs().parse_args(train_arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=train_args, train_func=chemprop.train.run_training)
    print(mean_score, std_score)

    predict_arguments = [
        '--test_path', f'{data_save_path}/{data_name}_infer_drugbank.csv',
        # '--features_generator', 'rdkit_2d_normalized',
        '--no_features_scaling',
        '--checkpoint_dir', f'{checkpoint_path}/full_data_{data_name}_checkpoints/',
        '--preds_path', f'{path}/predictions/all_data_infer_{data_name}_preds.csv',
        '--smiles_column', 'Smiles',
    ]

    if use_additional_features:
        predict_arguments += ['--features_path']
        for feature in additional_features:
            predict_arguments += [
                f'{data_save_path}/{data_name}_infer_drugbank_{str(feature)}.csv'
            ]

    predict_args = chemprop.args.PredictArgs().parse_args(predict_arguments)
    preds = chemprop.train.make_predictions(args=predict_args)
    
    preds_name = f'all_data_infer_{data_name}_preds'
    
    if use_additional_features:
        new_name = preds_name
        for i, feature in enumerate(additional_features):
            new_name += '_' + str(feature)
        os.rename(f'{path}/predictions/all_data_infer_{data_name}_preds.csv',
                f'{path}/predictions/{new_name}.csv')
    
    print('enriching data')
    preds_path = f'{path}/predictions/{new_name}.csv'
    df = pd.read_csv(preds_path)
    enriched_df = enrich_data(df, df.columns[0], './data/DrugBankReleases', '5.1.8')
    print(enriched_df.shape)
    enriched_df.to_csv(f'{path}/predictions/{new_name}_w_drug_bank_info.csv', index=False)

if __name__ == "__main__":
    main()