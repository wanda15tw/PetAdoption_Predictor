from src.d01_data.read_data import *
from src.d02_intermediate.preprocessing import *
from src.d03_processing.build_features import *
from src.d04_modeling.train_model import *

if __name__ == "__main__":

	# read all tables in Pandas DataFrame
	train, test, breed_labels, color_labels, state_labels = read_all_tables() 
	print(train.shape, test.shape)

	# preprocessing
	## Categorical feature preprocessing to String/Categorical
	train = feature_Num2Cate(train)
	test = feature_Num2Cate(test)
	print(train.shape, test.shape)

	## Label Mapping 
	mapbreed, mapcolor, mapstate = df_to_dict(breed_labels, 'BreedID', 'BreedName'), \
									df_to_dict(color_labels, 'ColorID', 'ColorName'), \
									df_to_dict(state_labels, 'StateID', 'StateName')
	train, test = feature_labels_mapping(train, (mapbreed, mapcolor, mapstate)), \
					feature_labels_mapping(test, (mapbreed, mapcolor, mapstate))
	print(train.shape, test.shape)

	# New Features
	## Google Sentiment API
	train, test = make_sentiment_df(train), make_sentiment_df(test, data_type='test')
	print(train.shape, test.shape)

	## make a copy in hdf at local for quick access
	train.to_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate.h5', key='train', format='table')
	test.to_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate.h5', key='test', format='table')


	## load train & test from intermediate
	train = pd.read_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate.h5', key='train')
	test = pd.read_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate.h5', key='test')


	# More New Features
	## topic modeling (tf, tfidf vectorizer & lda, nmf models are trained on train dataset only)
	train = topic_modeling(train)
	print(train.shape)
	train.to_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate2.h5', key='train', format='table')

	train = pd.read_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate2.h5', key='train')

	# add new features to the training dataset
	train = add_new_features(train)
	print(train.shape)

	train = processing(train)
	print(train.shape)
	train = pd.to_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/03_processed/processed.h5', key='train')


    # Modeling
    ## Model selection
    ## Metric: cohen_kappa_score
    


    ## choose the best out of DecisionTreeClassifier, SVC, KNeighborsClassifier, KNN, GaussianNB, RandomForest, XGB