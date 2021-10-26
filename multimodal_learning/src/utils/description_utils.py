import os

import pandas as pd

from src.persistant.readers.xml_reader.drugbank_reader import DrugReader
from src.utils.target_att import contains_cancer_word


def get_drug_details(path: str, version: str):
    #Reads the drug names
    reader = DrugReader()
    drug_bank = reader.get_drug_data(path, version)
    # d2d_releases_r1 = d2d_releases_reader()
    # drug_reader = d2d_releases_r1.read_release("5.1.8")
    drug_ids = sorted(list(drug_bank.id_to_drug.keys()))
    drug_names = pd.DataFrame({'drugBank_id': drug_ids,
                               'Drug_name': [drug_bank.id_to_drug[id_].name for id_ in drug_ids],
                              'drug_approved': ['approved' in drug_bank.id_to_drug[id_].groups for id_ in drug_ids],
                               'drug_withdrawn': ['withdrawn' in drug_bank.id_to_drug[id_].groups for id_ in drug_ids]
    })
    drug_names = drug_names.set_index('drugBank_id')
    return drug_names

def add_drug_names(df:pd.DataFrame, path: str, version: str, drugbank_id_col = 'drugBank_id'):
    ans = pd.merge(df, get_drug_details(path, version), how='left', left_on=drugbank_id_col, right_index=True)
    cols = list(ans.columns)
    cols.remove('Drug_name')
    cols.remove('drug_approved')
    cols.remove('drug_withdrawn')

    ans = ans[['Drug_name','drug_approved','drug_withdrawn']+cols]
    return ans

def add_clinical_trails_details(df: pd.DataFrame, path: str, drugbank_id_col = 'drugBank_id'):
    # clinical_trials_data = pd.read_csv(r'C:\Users\Administrator\Google Drive\לימודים\PHD\drug target prediction\data\num_trials_similar_drugs.csv',index_col=0)
    clinical_trials_data = pd.read_csv(path, index_col=0)
    clinical_trials_data = clinical_trials_data[~clinical_trials_data.index.duplicated(keep='first')]

    #clinical_trials_data = clinical_trials_data.groupby('drugbank_id').num_trials.sum()
    output = pd.merge(df, clinical_trials_data, how='left', left_on=drugbank_id_col, right_index=True)
    return output


def add_pre_clinical_trails_details(df: pd.DataFrame, path: str, drugbank_id_col = 'drugBank_id'):
    preclinical_trials_data = pd.read_csv(path, index_col='drugbank_id')
    # preclinical_trials_data = pd.read_csv(r'C:\Users\Administrator\Google Drive\לימודים\PHD\drug target prediction\data\drugbank_id_preclinical_filtered.csv',index_col='drugbank_id')
    
    del preclinical_trials_data['pubchem_id']
    del preclinical_trials_data['Unnamed: 0']
    del preclinical_trials_data['name']
    preclinical_trials_data = preclinical_trials_data[~preclinical_trials_data.index.duplicated(keep='first')]
    output = pd.merge(df, preclinical_trials_data, how='left', left_on=drugbank_id_col, right_index=True)
    return output


def add_mesh_antinoeoplastic(df: pd.DataFrame, path: str, drugbank_id_col = 'drugBank_id'):
    mesh_data = pd.read_csv(path, index_col='drugbank_id')
    # mesh_data = pd.read_csv(r'C:\Users\Administrator\PycharmProjects\multimodal_learning\output\targets\drugbank_id_antineoplastic_mesh.csv',index_col='drugbank_id')
    del mesh_data['Meshid']
    del mesh_data['name']
    del mesh_data['pubchem_id']
    mesh_data = mesh_data[~mesh_data.index.duplicated(keep='first')]
    mesh_data['mesh_antineoplastic']=1
    output = pd.merge(df, mesh_data, how='left', left_on=drugbank_id_col, right_index=True)
    return output



def add_description(df, path: str, version: str, drugbank_id_col='drugBank_id'):
    reader = DrugReader()
    drug_bank = reader.get_drug_data(path, version)
    drug_ids = sorted(list(drug_bank.id_to_drug.keys()))
    desc = pd.DataFrame({'drugBank_id': drug_ids,
                         'description':[drug_bank.id_to_drug[id_].description for id_ in drug_ids]})
    desc = desc.set_index('drugBank_id')
    desc['cancer_desc'] = desc.description.apply(contains_cancer_word)

    output = pd.merge(df, desc, how='left', left_on=drugbank_id_col, right_index=True)
    return output


def enrich_data(data, drugBank_col_name, drug_bank_path, version):

    output = add_drug_names(data, drug_bank_path, version, drugBank_col_name)
    output = add_mesh_antinoeoplastic(output, './data/DTI/cancer_enrich/drugbank_id_antineoplastic_mesh.csv', drugBank_col_name)
    output = add_clinical_trails_details(output, './data/DTI/cancer_enrich/num_trials_similar_drugs.csv', drugBank_col_name)
    output = add_pre_clinical_trails_details(output, './data/DTI/cancer_enrich/drugbank_id_preclinical_filtered.csv', drugBank_col_name)
    output = add_description(output, drug_bank_path, version, drugBank_col_name)
    
    return output


def enrich_predicitons(preds_save_path, feature_names):

    last_dot_idx = preds_save_path.rfind('.')
    new_name = preds_save_path[:last_dot_idx]

    new_name += feature_names
    os.rename(preds_save_path, new_name + '.csv')
    
    print('enriching data')
    df = pd.read_csv(new_name + '.csv')
    enriched_df = enrich_data(df, df.columns[0], './data/DrugBankReleases', '5.1.8')
    print(enriched_df.shape)

    enriched_df.to_csv(new_name + "w_drugbank_info" + '.csv', index=False)

if __name__ == "__main__":
    disease = 'cancer_clinical_trials'
    path = os.path.join('output','targets',disease,'predictions')
    for data_name in [
        '0_labels_preds.csv',
            '1_labels_preds.csv',
            '2_labels_preds.csv',
            'all_data_infer_labels_preds.csv',
                ]:
        try:
            data = pd.read_csv(os.path.join(path,data_name))
            drugBank_col_name = data.columns[0]
            print('drugbank col namee', drugBank_col_name)
            output = enrich_data(data,drugBank_col_name)
            output.to_csv(os.path.join(path,'w_details_'+data_name),index=False)
        except:
            print('failed',data_name)
    if True:
        data = pd.read_csv(r'C:\Users\Administrator\PycharmProjects\multimodal_learning\output\targets\cancer_clinical_trials\raw\labels.csv')
        drugBank_col_name = data.columns[0]
        print('drugbank col namee', drugBank_col_name)
        output = enrich_data(data,drugBank_col_name)
        output.to_csv(r'C:\Users\Administrator\PycharmProjects\multimodal_learning\output\targets\cancer_clinical_trials\raw\labels_w_details.csv',index=False)