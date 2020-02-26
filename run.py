from src.d01_data.read_data import *
from src.d02_intermediate.preprocessing import *
from src.d03_processing.build_features import *
from src.d04_modeling.train_model import *
from src.d04_modeling.predict_model import *
import pandas as pd
import pickle, os


HOME = '/Users/wandajuan/Documents/2020/08 PetAdoption'
RAW_DATA_DIR = 'data/01_raw/'
INT_DATA_DIR = 'data/02_intermediate'
PRO_DATA_DIR = 'data/03_processed'
MODEL_DIR = 'data/04_models'
# path = os.path.join(HOME, RAW_DATA_DIR)
REPORT_DIR = 'data/06_reporting'

if __name__ == "__main__":

	# read all tables in Pandas DataFrame
	train, test, breed_labels, color_labels, state_labels = read_all_tables() 
	print(train.shape, test.shape)

	# preprocessing
	## Categorical feature preprocessing to String/Categorical
	train = feature_Num2Cate(train)
	test = feature_Num2Cate(test)


	## Label Mapping 
	mapbreed, mapcolor, mapstate = df_to_dict(breed_labels, 'BreedID', 'BreedName'), \
									df_to_dict(color_labels, 'ColorID', 'ColorName'), \
									df_to_dict(state_labels, 'StateID', 'StateName')
	train, test = feature_labels_mapping(train, (mapbreed, mapcolor, mapstate)), \
					feature_labels_mapping(test, (mapbreed, mapcolor, mapstate))
	print('Done initial preprocessing.')
	print(train.shape, test.shape)


	# New Features
	## Google Sentiment API
	train, test = make_sentiment_df(train), make_sentiment_df(test, data_type='test')
	print('Added sentiment columns.')
	print(train.shape, test.shape)

	## make a copy in hdf at local for quick access
	# train.to_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate.h5', key='train', format='table')
	# test.to_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate.h5', key='test', format='table')


	## load train & test from intermediate
	# train = pd.read_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate.h5', key='train')
	# test = pd.read_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate.h5', key='test')


	# More New Features
	## topic modeling (tf, tfidf vectorizer & lda, nmf models are trained on train dataset only)
	train_len = train.shape[0]
	df = train.append(test)
	df = topic_modeling(df)
	print('Done topic modeling on train & test.')

	train = df[:train_len]
	test = df[train_len:]
	test.drop('AdoptionSpeed', axis=1, inplace=True)
	print(train.shape, test.shape)
	# train.to_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate2.h5', key='train', format='table')

	# train = pd.read_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate2.h5', key='train')

	# add new features to the training dataset
	train, test = add_new_features(train), add_new_features(test)
	print(train.shape, test.shape)
	# train.to_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate3.h5', key='train')
	y_train = train['AdoptionSpeed']
	test_PetID = test['PetID']

	X_train, test = processing(train), processing(test)
	print(train.shape, test.shape)
	# train.to_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/03_processed/processed.h5', key='train')

	X_train['AdoptionSpeed'] = y_train
	train = X_train

	# Modeling
	## Model selection
	metrics, clfs, best_model, best_metrics = model_selection(train)
	# pickle.dump(best_model, open(os.path.join(HOME, MODEL_DIR, 'best_model2'), 'ab'))


	# make prediction by the best_model
	y_test = best_model.predict(test)

	# make submission
	submission = pd.DataFrame()
	submission['PetID'] = test_PetID
	submission['AdoptionSpeed'] = y_test

	submission.to_csv(os.path.join(HOME, REPORT_DIR, 'submission.csv'))
