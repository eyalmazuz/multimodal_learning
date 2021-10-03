from src.datasets.drug_interactions.dataset_builder import DatasetTypes
from src.features.drug_interactions.Char2VecFeature import Char2VecFeature
from src.features.drug_interactions.EmbeddingFeature import EmbeddingFeature
from src.features.drug_interactions.OneHotFeature import OneHotFeature
from src.features.drug_interactions.CNNFeature import CNNFeature

def get_features(dataset_type, **kwargs):
    features = []
    if dataset_type == DatasetTypes.AFMP:
        features = [EmbeddingFeature(**kwargs)]

    elif dataset_type == DatasetTypes.ONEHOT_SMILES:
        features = [OneHotFeature(**kwargs)]

    elif dataset_type == DatasetTypes.CHAR_2_VEC:
        features = [Char2VecFeature(**kwargs)]

    elif dataset_type == DatasetTypes.DEEP_SMILES:
        features = [OneHotFeature(**kwargs), CNNFeature(**kwargs)]

    return features