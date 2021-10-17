from bokeh.plotting import figure, show, ColumnDataSource, output_file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


from src.utils.utils import read_h5
from src.utils.description_utils import add_drug_names

def main():

    h5 = read_h5()
    df = h5['/df']
    h5.close()

    feature_df = pd.read_csv('./data/DTI/features/DDIFeature_256.csv', index_col=0)
    labels = pd.read_csv('./data/DTI/cancer_clinical_trials/processed/labels_training_set_w_drugbank_id.csv', index_col=0)
    print(feature_df.shape)
    pca = PCA(n_components=2)
    features_pca = pd.DataFrame(pca.fit_transform(feature_df))
    features_pca['drugBank_id'] = feature_df.index.tolist()
    features_pca = features_pca.set_index('drugBank_id')
    features_pca.rename(columns={0:'pca_1', 1:'pca_2'}, inplace=True)

    approved_drugs = df[df['Group: approved'] == True].index.tolist()
    features_pca = features_pca[features_pca.index.isin(approved_drugs)]
    drug_cancer_labels = [labels.loc[drug_id].cancer if drug_id in labels.index else 2 for drug_id in features_pca.index]
    
    def convert_caner_to_label(x):
        if x == 1:
            return "Anti-cancer"
        elif x == 0:
            return "No anti-cancer"
        elif x == 2:
            return "Unknown"

    features_pca['cancer'] = drug_cancer_labels
    features_pca['cancer'] = features_pca['cancer'].apply(convert_caner_to_label)

    features_pca = add_drug_names(features_pca, './data/DrugBankReleases', '5.1.8')


    cancer_df = features_pca[features_pca['cancer'] == "Anti-cancer"]
    no_cancer_df = features_pca[features_pca['cancer'] == "No anti-cancer"]
    unk_cancer_df = features_pca[features_pca['cancer'] == "Unknown"]

    output_file("toolbar.html")

    cancer_source = ColumnDataSource(data=dict(
        id_=cancer_df.index.tolist(),
        x=cancer_df.pca_1.tolist(),
        y=cancer_df.pca_2.tolist(),
        cancer=cancer_df.cancer.tolist(),
        name=cancer_df.Drug_name.tolist(),
        withdrawn=cancer_df.drug_withdrawn.tolist(),
    ))

    no_cancer_source = ColumnDataSource(data=dict(
        id_=no_cancer_df.index.tolist(),
        x=no_cancer_df.pca_1.tolist(),
        y=no_cancer_df.pca_2.tolist(),
        cancer=no_cancer_df.cancer.tolist(),
        name=no_cancer_df.Drug_name.tolist(),
        withdrawn=no_cancer_df.drug_withdrawn.tolist(),
    ))

    unk_source = ColumnDataSource(data=dict(
        id_=unk_cancer_df.index.tolist(),
        x=unk_cancer_df.pca_1.tolist(),
        y=unk_cancer_df.pca_2.tolist(),
        cancer=unk_cancer_df.cancer.tolist(),
        name=unk_cancer_df.Drug_name.tolist(),
        withdrawn=unk_cancer_df.drug_withdrawn.tolist(),
    ))

    TOOLTIPS = [
        ("cancer", "@cancer"),
        ("drugBank_id", "@id_"),
        ("Drug_name", "@name"),
        ("withdrawn", "@withdrawn"),
    ]
    p = figure(width=900, height=900, tooltips=TOOLTIPS, title="Mouse over the dots")

    p.scatter('x', 'y', size=7, source=cancer_source, line_color="orange", fill_color="orange")
    p.scatter('x', 'y', size=7, source=no_cancer_source, line_color="blue", fill_color="blue")
    p.scatter('x', 'y', size=7, source=unk_source, line_color="green", fill_color="green")


    show(p)

if __name__ == "__main__":
    main()