import pandas as pd 
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import pickle
import numpy as np
choiseTrain = False
testArry = False
if choiseTrain:
	train = pd.read_csv('train.csv')
	print(train.head())
	X = train.drop(['label'],axis=1)
	y = train['label']

	print('1')
	clf = LinearSVC(random_state=0)
	print('2')
	clf.fit(X, y)
	print('3')
	LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
		intercept_scaling=1, loss='squared_hinge', max_iter=1000,
		multi_class='ovr', penalty='12', random_state=0, tol=0.0001,
		verbose=0)
	print('4')
	print(clf.coef_)
	print(clf.intercept_)


	pickle.dump(clf, open( "save.p", "wb"))

elif testArry:
	test = pd.read_csv('test.csv')
	clf = pickle.load( open( "save.p", "rb" ) )
	result = clf.predict(test)
	np.save('my_array', result)

else:
	result = np.load('my_array.npy')
	# print('Size: ',result.size)
	# print('Shape: ',result.shape)
	df = pd.DataFrame(result)
	df.columns = ['Label']
	df.index += 1
	df.index.name = 'ImageId'

	print(df.head())

	df.to_csv('df')

	# print(df.apply(pd.Series.value_counts))





