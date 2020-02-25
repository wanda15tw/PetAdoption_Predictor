import pandas as pd
import os, time, json

def feature_Num2Cate(df):
	
	df['Type'] = df['Type'].replace({1:'Dog', 2:'Cat'})
	df['Gender'].replace({1:'Male', 2:'Female', 3: 'Mixed'}, inplace=True)
	df['MaturitySize'].replace({1 : 'Small', 2 : 'Medium', 3 : 'Large', 4 : 'Extra Large', 0 : 'Not Specified'}, inplace=True)
	df['FurLength'].replace({1 : 'Short', 2 : 'Medium', 3 : 'Long', 0 : 'Not Specified'}, inplace=True)
	df['Vaccinated'].replace({1 : 'Yes', 2 : 'No', 3 : 'Not Sure'}, inplace=True)
	df['Dewormed'].replace({1 : 'Yes', 2 : 'No', 3 : 'Not Sure'}, inplace=True)
	df['Sterilized'].replace({1 : 'Yes', 2 : 'No', 3 : 'Not Sure'}, inplace=True)
	df['Health'].replace({1 : 'Healthy', 2 : 'Minor Injury', 3 : 'Serious Injury', 0 : 'Not Specified'}, inplace=True)

	if 'AdoptionSpeed' in df.columns:
		# transform label to categorical 
		df['AdoptionSpeed'].replace({0: 'Same Day', 1: 'One Week', 2:'One Month', 3:'2 - 3 Month', 4:'No Adoption'}, inplace=True)
		df['AdoptionSpeed'] = pd.Categorical(df['AdoptionSpeed'], ordered=True, categories=['Same Day', 'One Week', 'One Month', '2 - 3 Month', 'No Adoption'])

	return df


def df_to_dict(label_df, index_name, value_name):
	return label_df.set_index(index_name).to_dict()[value_name]


def feature_labels_mapping(df, maps):
	mapbreed, mapcolor, mapstate = maps

	df['Breed1'] = df['Breed1'].replace(mapbreed)
	df['Breed2'] = df['Breed2'].replace(mapbreed)
	df['Color1'] = df['Color1'].replace(mapcolor)
	df['Color2'] = df['Color2'].replace(mapcolor)
	df['Color3'] = df['Color3'].replace(mapcolor)
	df['State'] = df['State'].replace(mapstate)

	df['Breed1'] = df['Breed1'].astype(str)
	df['Breed2'] = df['Breed2'].astype(str)
	df['Color1'] = df['Color1'].astype(str)
	df['Color2'] = df['Color2'].astype(str)
	df['Color3'] = df['Color3'].astype(str)
	df['State'] = df['State'].astype(str)

	return df

HOME = '/Users/wandajuan/Documents/2020/08 PetAdoption'
RAW_DATA_DIR = 'data/01_raw/'
path = os.path.join(HOME, RAW_DATA_DIR)

def make_sentiment_df(df, data_type='train'):
	sentiment_dir = os.path.join(path, data_type+'_sentiment')

	t0 = time.time()

	for file in os.listdir(sentiment_dir):
		petid = file.replace('.json', '')
		api = json.load(open(os.path.join(sentiment_dir, file)))
		df.loc[df['PetID']==petid, 'docsentiment-magnitude'] = api['documentSentiment']['magnitude']
		df.loc[df['PetID']==petid, 'docsentiment-score'] = api['documentSentiment']['score']
		df.loc[df['PetID']==petid, 'top entity'] = api['entities'][0]['name'] if len(api['entities'])>0 else None
	print(f'finish parsing sentiments for \"{data_type}\" in {time.time()-t0}')
	return df


if __name__ == "__main__":
	pass
