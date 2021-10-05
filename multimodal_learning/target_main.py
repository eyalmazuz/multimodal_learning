import os
import shutil

import chemprop
import pandas as pd

from src.persistant.readers.db_reader.db_reader import get_h5_data
from src.preprocess.chemprop_preprocessor import preprocess_data
from src.utils.utils import read_h5
from src.utils.description_utils import enrich_data
from src.persistant.readers.xml_reader.drugbank_reader import DrugReader

def main():
    version = '5.1.8'
    modalities_path = './data/DTI/h5/'

    path = './data/DTI/cancer_clinical_trials'
    checkpoint_path = f'{path}/cps'
    data_save_path = f'{path}/processed'

    if not os.path.exists(f'{modalities_path}/modalities_dict_{version}.h5'):
        get_h5_data(version=version, save_path=modalities_path)

    if not os.path.exists(f'{path}/processed/'):
        preprocess_data(data_path=f'{path}/raw',
                        modalities_path='./data/DTI/h5/',
                        version=version,
                        save_path=f'{path}/processed')

    data_name = 'labels'
    ensemble_size = '2'
    num_folds = '10'

    # if os.path.exists(f'{checkpoint_path}/evaluation_{data_name}/checkponts'):
    #     shutil.rmtree(f'{checkpoint_path}/evaluation_{data_name}/checkponts')
    
    # eval_arguments = [
    #     '--data_path', f'{data_save_path}/{data_name}_training_set.csv',
    #     '--features_generator', 'rdkit_2d_normalized',
    #     '--no_features_scaling',
    #     '--dataset_type', 'classification',
    #     '--num_workers', '8',
    #     '--extra_metrics', 'prc-auc',
    #     '--split_type', 'scaffold_balanced',
    #     '--split_sizes', '0.7', '0.1', '0.2',
    #     '--num_folds', f'{num_folds}',
    #     '--save_dir', f'{checkpoint_path}/evaluation_{data_name}_checkpoints',
    #     '--smiles_column', 'Smiles',
    #     '--epochs', '30',
    #     '--ensemble_size', f'{ensemble_size}',
    #     '--save_preds',
    #     '--save_smiles_splits',
    #     '--config_path', f'./data/DTI/jsons/full_data_hyperparams_w_rkdit.json'
    # ]

    # eval_args = chemprop.args.TrainArgs().parse_args(eval_arguments)
    # mean_score, std_score = chemprop.train.cross_validate(args=eval_args, train_func=chemprop.train.run_training)

    # if os.path.exists(f'{checkpoint_path}/full_data_{data_name}_checkpoints'):
    #     print('here')
    #     shutil.rmtree(f'{checkpoint_path}/full_data_{data_name}_checkpoints')
        
    # train_arguments = [
    #     '--data_path', f'{data_save_path}/{data_name}_training_set.csv',
    #     '--features_generator', 'rdkit_2d_normalized',
    #     '--no_features_scaling',
    #     '--dataset_type', 'classification',
    #     '--num_workers', '8',
    #     '--extra_metrics', 'prc-auc',
    #     '--split_type', 'scaffold_balanced',
    #     '--split_sizes', '0.9', '0.1', '0.0',
    #     '--save_dir', f'{checkpoint_path}/full_data_{data_name}_checkpoints',
    #     '--smiles_column', 'Smiles',
    #     '--epochs', '30',
    #     '--ensemble_size', f'{ensemble_size}',
    #     '--save_preds',
    #     '--save_smiles_splits',
    #     '--config_path', f'./data/DTI/jsons/full_data_hyperparams_w_rkdit.json'
    # ]

    # train_args = chemprop.args.TrainArgs().parse_args(train_arguments)
    # mean_score, std_score = chemprop.train.cross_validate(args=train_args, train_func=chemprop.train.run_training)
    # print(mean_score, std_score)

    # predict_arguments = [
    #     '--test_path', f'{data_save_path}/{data_name}_infer_drugbank.csv',
    #     '--features_generator', 'rdkit_2d_normalized',
    #     '--no_features_scaling',
    #     '--checkpoint_dir', f'{checkpoint_path}/full_data_{data_name}_checkpoints/',
    #     '--preds_path', f'{path}/predictions/all_data_infer_{data_name}_preds.csv',
    #     '--smiles_column', 'Smiles',
    # ]

    # predict_args = chemprop.args.PredictArgs().parse_args(predict_arguments)
    # preds = chemprop.train.make_predictions(args=predict_args)

    # reader = DrugReader()
    # drug_bank = reader.get_drug_data('./data/DrugBankReleases', '5.1.8')

    # keys = list(drug_bank.id_to_drug.keys())
    # for key in keys[:10]:
    #     print(drug_bank.id_to_drug[key].description)
    preds_path = f'{path}/predictions/all_data_infer_{data_name}_preds.csv'
    df = pd.read_csv(preds_path)
    enriched_df = enrich_data(df, df.columns[0], './data/DrugBankReleases', '5.1.8')
    print(enriched_df.shape)
    enriched_df.to_csv(f'{path}/predictions/all_data_infer_{data_name}_preds_w_drug_bank_info.csv', index=False)

if __name__ == "__main__":
    main()