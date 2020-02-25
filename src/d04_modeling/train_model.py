from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer
import time, pickle

def get_score(y_true, y_pred):
	return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def train_to_get_score(clf, X_train, y_train, X_test, y_test):
	clf.fit(X_train, y_train)
	train_acc = clf.score(X_train, y_train)
	train_score = get_score(y_train, clf.predict(X_train))
	test_acc = clf.score(X_test, y_test)
	test_score = get_score(y_test, clf.predict(X_test))

	print(f'Model: {clf}')
	print('train accuarcy: {0:.4f}'.format(train_acc))
	print('train score: {0:.4f}'.format(train_score))
	print('test accuarcy: {0:.4f}'.format(test_acc))
	print('test accuarcy: {0:.4f}'.format(test_score))

	return train_acc, train_score, test_acc, test_score


def model_selection(df):
	y = df['AdoptionSpeed']
	X = df.drop('AdoptionSpeed', axis = 1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

	from sklearn.tree import DecisionTreeClassifier
	from sklearn.svm import SVC
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.ensemble import RandomForestClassifier
	from xgboost import XGBClassifier

	models = ['DecisionTreeClassifier', 'SVC', 'KNeighborsClassifier', 'GaussianNB', 'RandomForest', 'XBG']
	clfs = [DecisionTreeClassifier(), 
			SVC(kernel='linear', C=1),
			KNeighborsClassifier(n_neighbors=5),
			GaussianNB(),
			RandomForestClassifier(n_estimators=10, criterion='entropy'),
			XGBClassifier(objective='multi:softprob', learning_rate=0.1, n_estimator=500)
			]

	metrics = {}

	for i, model in enumerate(models):
		print(f'training {model}')
		t0 = time.time()
		metrics[model] = train_to_get_score(clf=clfs[i], X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		print(f'finish in {time.time()-t0}\n')



	def GridSearch(X_train, y_train, clf):

		params = {'n_estimators':[500, 1000, 1300], 'criterion':['gini', 'entropy'] }
		kappa_score = make_scorer(cohen_kappa_score, weights='quadratic')
		grid_obj = GridSearchCV(clf, params, scoring=kappa_score)
		grid_obj.fit(X_train, y_train)
		return grid_obj.best_estimator_

	t0 = time.time()
	best_model = GridSearch(X_train, y_train, clfs[4]) # manually choose RandomForest as the best model to fine tune
	print(f'best model found: {best_model} and fine tuning done in {time.time()-t0}')
	best_metrics = train_to_get_score(best_model, X_train, y_train, X_test, y_test)
	return metrics, clfs, best_model, best_metrics




