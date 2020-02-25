import pickle, os

HOME = '/Users/wandajuan/Documents/2020/08 PetAdoption'
MODEL_DIR = 'data/04_models'
path = os.path.join(HOME, MODEL_DIR)


def input_processing(df):
	pass


def predict(X, model_path='best_model_rf'):
	model = pickle.load(open(os.path.join(path, model_path), 'rb'))
	y = model.predict(X)
	return y
