import os
import pandas as pd
from pandas.core.indexes.base import Index
from sklearn.decomposition import PCA

from src.features.drug_interactions.get_features import get_features
from src.persistant.readers.xml_reader.drugbank_reader import DrugReader
from src.preprocess.xml_preprocessor import DrugPreprocessor
from src.datasets.drug_interactions.dataset_builder import get_dataset, DatasetTypes
from src.training.train import Trainer
from src.models.drug_interactions.model_builder import get_model


class DDIFeature():

    def __init__(self, emb_dim: int=64, features_path: str=None, str=None, **kwargs):
        # self.target_df = target_df
        self.emb_dim = emb_dim
        self.features_path = features_path
        self.old_version = kwargs['old_version']
        self.drugbank_path = kwargs['drugbank_path']
        self.kwargs = kwargs

    def __repr__(self,):
        return f"DDIFeature_{self.emb_dim}"


    def __call__(self):
        if not os.path.exists(f'{self.features_path}/{str(self)}.csv'):
            
            # Train and get AFMP embs
            embbeding, ids = self.get_emb()
            # Create feature DF
            features_ddi = pd.DataFrame(columns=['drugBank_id'] + [i for i in range(self.emb_dim)])            
            for i in range(len(ids)):
                id_ = ids[i]
                vec = embbeding[i]
                d = {**{'drugBank_id': id_}, **dict(zip(range(len(vec)), vec.tolist()))}
                df = df.append(d, ignore_index=True)

            features_ddi = features_ddi.set_index('drugBank_id')

            # Save the Features
            if not os.path.exists(f'{self.features_path}'):
                os.makedirs(f'{self.features_path}/', exist_ok=True)

            features_ddi.to_csv(f'{self.features_path}/{str(self)}.csv', index=True)
        else:
            features_ddi = pd.read_csv(f'{self.features_path}/{str(self)}.csv', index_col=0)

        return features_ddi

    def get_emb(self):
        reader = DrugReader()

        old_drug_bank = reader.get_drug_data(self.drugbank_path, self.old_version)
        preprocessor =  DrugPreprocessor()

        old_drug_bank = preprocessor.get_preprocessed_drug_bank(old_drug_bank, f'{self.drugbank_path}/{self.old_version}')


        dataset_type_str = str(DatasetTypes.AFMP).split(".")[1]
        feature_config = {
            # CNN features
            "atom_size": 300,
            "atom_info": 21,
            "struct_info": 21,
            # Char2Vec feature
            "embedding_size": self.emb_dim,
            "window": 5,
            "min_count": 1,
            "workers": 8,
            "epochs": 5,
        }

        features = get_features(DatasetTypes.AFMP, **feature_config)

        (train_dataset, validation_dataset, metadata) = get_dataset(old_drug_bank,
                                                                    new_drug_bank=None,
                                                                    feature_list=features,
                                                                    **self.kwargs)

        metadata['embedding_size'] = self.emb_dim
        model = get_model(DatasetTypes.AFMP, **metadata)

        trainer = Trainer(epoch_sample=False, balance=False)

        trainer.train(model, train_dataset, validation_dataset, epochs=3, dataset_type=dataset_type_str)

        drug_emb = model.drug_embedding.get_weights()

        return drug_emb, train_dataset.features['EmbeddingFeature']