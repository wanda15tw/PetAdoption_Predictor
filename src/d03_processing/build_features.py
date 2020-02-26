import json, os, time, re
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
import pandas as pd


HOME = '/Users/wandajuan/Documents/2020/08 PetAdoption'
INT_DATA_DIR = 'data/02_intermediate/'
path = os.path.join(HOME, INT_DATA_DIR)





def topic_modeling(df):

	t0 = time.time()

	# prepare corpus
	df['Desc_rm_stop'] = df['Description'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in stopwords.words('english')]))
	documents = [doc for doc in df['Desc_rm_stop']]

	no_topics = 10
	no_features = 1000

	# LDA uses Count Vectorizer
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
	tf = tf_vectorizer.fit_transform(documents)
	tf_feature_names = tf_vectorizer.get_feature_names()
	lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
	lda_output = lda.transform(tf)

	pickle.dump(tf_vectorizer, open(os.path.join(path, 'tf'), 'ab'))
	pickle.dump(lda, open(os.path.join(path, 'lda_model'), 'ab'))

	df['lda_topic'] = np.argmax(lda_output, axis=1)
	
	# NMF uses tf-idf
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
	tfidf = tfidf_vectorizer.fit_transform(documents)
	tfidf_feature_names = tfidf_vectorizer.get_feature_names()

	nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
	nmf_output = nmf.transform(tfidf)

	pickle.dump(tfidf_vectorizer, open(os.path.join(path, 'tfidf'), 'ab'))
	pickle.dump(nmf, open(os.path.join(path, 'nmf_model'), 'ab'))

	df['nmf_output'] = np.argmax(nmf_output, axis=1)

	print(f'finish topic modeling (LDA & NMF) in {time.time() - t0} & 3 features are created {df.columns[-3:].values}')

	return df


def add_new_features(df):

	col0 = df.columns
	# word count in `description`
	df['Desc_WC'] = df['Description'].apply(lambda x: re.sub(r'[^\d\w\/]+\s*', ' ', str(x).lower()))
	df['Desc_WC'] = df['Desc_WC'].apply(lambda x: len(x.split()))

	# clean up Breed1 & Breed2 so that Breed1 has no '0'
	x = df.loc[df['Breed1']=='0', 'Breed2']
	df.loc[df['Breed1']=='0', 'Breed2'] = '0'
	df.loc[df['Breed1']=='0', 'Breed1'] = x

	# Breed count
	df['breed_cnt'] = df[['Breed1', 'Breed2']].apply(lambda df: 1 if df['Breed2']=='0' else 2, axis=1)

	# color count
	df['color_cnt'] = (df[['Color1', 'Color2', 'Color3']]!='0').sum(axis=1)


	# top 20 breeds vs rare
	top20breed = df['Breed1'].value_counts().iloc[:20].index
	df['top_breed'] = df['Breed1'].apply(lambda x: x if x in top20breed else 'rare')


	# common top entities
	df['top entity'] = df['top entity'].apply(lambda x: str(x).lower())
	df['top entity'].replace({'dogs': 'dog', 'cats':'cat', 'kitten':'cat', 'kittens':'cat', 'kitty':'cat',\
                             'puppy':'dog', 'puppies':'dog', 'pup':'dog', 'pups':'dog'}, inplace=True)
	top30entity = df['top entity'].value_counts()[:30].index
	df['top entity'] = df['top entity'].apply(lambda x: x if x in top30entity else 'rare')

	# clean up 'gender'
	df['Gender'] = df['Gender'].apply(lambda x: 'Other' if x in ['Mixed', 0] else x)


	print(f'new columns added: {[col for col in df.columns if col not in col0]}')

	return df




def encoding(df):

	# One-Hot encoder
	cols = ['Type', 'Gender', 'Color1', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health']
	df = pd.get_dummies(df, columns=cols)


	# Label encoder
	from sklearn.preprocessing import LabelEncoder
	le = LabelEncoder()
	cols = ['top_breed', 'State', 'top entity']

	for col in cols:
		df.loc[:, col] = le.fit_transform(df.loc[:, col].values.reshape(-1, 1))

	return df


def scaling(df):

	from sklearn.preprocessing import MinMaxScaler

	scaler = MinMaxScaler(feature_range=(-1, 1))
	cols = df.columns[df.max(axis=0)>1]#['Age', 'breed_cnt', 'color_cnt', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'Desc_WC']
	cols = set(cols) - {'AdoptionSpeed'}
	for col in cols:
		df.loc[:, col] = scaler.fit_transform(df.loc[:, col].values.reshape(-1, 1))

	return df


def processing(df):
    # Processing to build features
    ## feature selection
    df = df.loc[:, ['Type', 'Age', 'top_breed', 'breed_cnt', 'Gender', 'Color1', 'color_cnt', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'VideoAmt', 'PhotoAmt', 'docsentiment-score', 'top entity', 
       'lda_topic', 'nmf_output', 'Desc_WC']]

    ## Imputation for missing values
    df.loc[df['docsentiment-score'].isnull(), 'docsentiment-score'] = 0
    

    ## Encoding
    df = encoding(df)
    ## Scaling for numerical features
    df = scaling(df)

    return df



if __name__ == "__main__":
	pass