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
from src.utils.utils import convert_to_IC50, read_h5, drug_names_to_drugBank
from src.utils.target_att import target_att_pancreas, nih_pancreas_drug_names 

def main():

    cur_date = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    TASK = 'yeast'

    drugbank_version = '5.1.8'
    modalities_path = './data/DTI/h5'
    cancer_path = './data/DTI/cancer_clinical_trials'
    features_path = './data/DTI/features'
    
    checkpoint_path = f'./data/DTI/yeast/cps'
    train_path = f'./data/DTI/yeast/{cur_date}_targets.csv'
    test_path = f'./data/DTI/yeast/test.csv'

    preds_save_path = f'./data/DTI/yeast/predictions/{cur_date}_predictions.csv'

    ensemble_size = '2'
    num_folds = '10'
    
    use_additional_features = False

    if TASK == "IC50": 
        print('Running IC50')
        df = get_chembl_data('./data/chembl_29_sqlite/chembl_29.db',

                        [
                        'CHEMBL2189121', # KRAS
                         'CHEMBL4096', # TP53
                         'CHEMBL3217396', # JAG1
                         'CHEMBL227', # AGTP1
                         'CHEMBL5606', # STK11
                         'CHEMBL1783', # VEGFA
                         'CHEMBL4295916', # NDGR1
                        #  'CHEMBL1075094', # NFE2L2
                         'CHEMBL3712919', # GPNBM
                         'CHEMBL2346489', # KITLG
                         'CHEMBL4164', # TRP53
                         'CHEMBL3407319', # NOTCH3
                         'CHEMBL5460', # HSPA1A
                         'CHEMBL3712878', # MSLN
                         'CHEMBL1993', # DNMT1
                        #  'CHEMBL3407321', # NOTCH4
                         'CHEMBL3885585', # HSPA1B
                         'CHEMBL3734643', # HES1
                         'CHEMBL1795148', # ARG2
                         'CHEMBL2146346' # NOTCH1
                        ],
                        ['IC50'],)

        if not os.path.exists(train_path):
            prepare_chembl_data(df, train_path)
    
    elif TASK == "cancer":
        print('Running cacner prediction')
        prepare_cancer_data(version=drugbank_version,
                              modalities_path=modalities_path,
                              path=cancer_path)
                            

    elif TASK == "yeast":
        print('Running yeast')

        TOP_K = 10

        print('Loading data')
        df = pd.read_csv('./data/DTI/yeast/raw_yeast.tsv', sep='\t')

        print('Extracting top targets')
        target_count = df[['Gene_Symbol', 'Activity_Flag']].groupby('Gene_Symbol').count()
        target_count = target_count.sort_values('Activity_Flag', ascending=False)
        print(target_count.head(10))
        top_targets = target_count.head(TOP_K).index.tolist()

        # raise ValueError


        df = df[df['Gene_Symbol'].isin(top_targets)]
        # df = df[~df['Ambit_InchiKey'].isin(top_targets)]
        # print(df1.shape)
        # print(df2.shape)
        df['Activity_Flag'] = 1

        print('Creating pivot table')
        pt = pd.pivot_table(df, values='Activity_Flag', index='SMILES', columns='Gene_Symbol', fill_value=0)
        # pt = pt[top_targets]
        pt.index.rename('Smiles', inplace=True)

        print(pt.shape)

        print('Saving data')
        pt.to_csv(train_path, index=True)

    task_eval_arguments, task_train_arguments = get_task_config(TASK)

    features = [
        # TargetPCAFeature,
        # DDIFeature
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

    if use_additional_features:
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
    
    if TASK == "IC50":
        df = pd.read_csv(preds_save_path, index_col=0)
        target_columns = df.columns.tolist()
        target_columns.remove('Smiles')
        for column in target_columns:
            df[column] = df[column].apply(convert_to_IC50)

        df.to_csv(preds_save_path, index=True)
    
    enrich_predicitons(preds_save_path, features_names)

if __name__ == "__main__":
    main()