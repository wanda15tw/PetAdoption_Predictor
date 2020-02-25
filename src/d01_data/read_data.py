# read data: train, test
import os
import pandas as pd
import time

HOME = '/Users/wandajuan/Documents/2020/08 PetAdoption'
RAW_DATA_DIR = 'data/01_raw/'
path = os.path.join(HOME, RAW_DATA_DIR)


def read_to_pd(df_name):
	return pd.read_csv(df_name+'.csv')

def read_all_tables():

	t0 = time.time()
	train = read_to_pd(os.path.join(path, 'train'))
	test = read_to_pd(os.path.join(path, 'test'))
	assert train.shape == (14993, 24)
	assert test.shape == (3972, 23)
	breed_labels = read_to_pd(os.path.join(path, 'breed_labels'))
	color_labels = read_to_pd(os.path.join(path, 'color_labels'))
	state_labels = read_to_pd(os.path.join(path, 'state_labels'))
	assert state_labels.shape == (15, 2)
	assert breed_labels.shape == (307, 3)
	assert color_labels.shape == (7, 2)

	print(f"read all tables in {time.time()-t0}")
	return train, test, breed_labels, color_labels, state_labels

if __name__ == '__main__':
	read_all_tables()