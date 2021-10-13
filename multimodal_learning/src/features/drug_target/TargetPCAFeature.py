import os
import pandas as pd
from sklearn.decomposition import PCA

from src.utils.utils import read_h5


class TargetPCAFeature():

    def __init__(self, pca_dim: int=64, features_path: str=None, modalities_path: str=None, **kwargs):
        # self.target_df = target_df
        self.pca_dim = pca_dim
        self.features_path = features_path
        self.modalities_path = modalities_path

    def __repr__(self,):
        return f"TargetPCAFeature_{self.pca_dim}"


    def __call__(self):
        if not os.path.exists(f'{self.features_path}/{str(self)}.csv'):
            h5 = read_h5(self.modalities_path)
            df = h5['/df']
            h5.close()
            target_df_columns = [column for column in df.columns if 'Target:' in column]
            target_df = df[target_df_columns]
            pca = PCA(n_components=self.pca_dim)
            

            #Removing the smiles column and any drug that doesn't have target
            nans = target_df[target_df.isna().any(axis=1)]
            features_df = target_df[~target_df.index.isin(nans.index)]
            features_pca = pd.DataFrame(pca.fit_transform(features_df))
            drugBank_id = target_df[~target_df.index.isin(nans.index)].index
            features_pca['drugBank_id'] = drugBank_id
            features_pca = features_pca.set_index('drugBank_id')

            # Save the Features
            if not os.path.exists(f'{self.features_path}'):
                os.makedirs(f'{self.features_path}/', exist_ok=True)

            features_pca.to_csv(f'{self.features_path}/{str(self)}.csv', index=True)
        else:
            features_pca = pd.read_csv(f'{self.features_path}/{str(self)}.csv', index_col=0)

        return features_pca