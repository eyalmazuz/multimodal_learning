import os

from src.persistant.readers.db_reader.db_reader import get_h5_data
from src.preprocess.chemprop_preprocessor import preprocess_data


def main():
	version = '5.1.8'
	save_path = './data/targets/h5/'

	if not os.path.exists(f'{save_path}/modalities_dict_{version}.h5'):
		get_h5_data(version=version, save_path=save_path)

	preprocess_data(data_path='./data/targets/cancer_clinical_trials/raw',
					modalities_path='./data/targets/h5/',
					version=version,
					save_path='./data/targets/cancer_clinical_trials/')

if __name__ == "__main__":
	main()