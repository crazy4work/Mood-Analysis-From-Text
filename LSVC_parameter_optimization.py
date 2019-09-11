import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import itertools
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.externals import joblib

def draw_confusionmatrix(cm,classes,title,cmap=plt.cm.Blues):
	#creating a figure object
	plt.figure()
	#connecting cm and figure
	plt.imshow(cm,cmap=cmap)
	#giving title to the plot
	plt.title(title)
	plt.colorbar()
	#no. of marking of the axes
	tick_mark=np.arange(len(classes))
	#marking x axis)
	plt.xticks(tick_mark,classes)
	#marking y axis
	plt.yticks(tick_mark,classes)
	#writing the confusion matrix
	for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
		plt.text(j,i,format(cm[i][j],'d'),horizontalalignment='center')
	plt.ylabel('true label')
	plt.xlabel('predicted label')
	plt.savefig('cm_lr.png')


def main():
	train_dataset=pd.read_csv("train.csv",usecols=['sentiment','content'])
	test_dataset=pd.read_csv("test.csv",usecols=['sentiment','content'])

	#classes=dict(Counter(train_dataset['sentiment'])).keys()
	classes=["happy","sad","fear/surprise","angry/disgust"]

	vect=TfidfVectorizer(
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 2),stop_words="english",
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)


	dtm_train_x=vect.fit_transform(train_dataset['content'])
	print(len(vect.get_feature_names()))
	dtm_test_x=vect.transform(test_dataset['content'])

	#svm paramtres tuning
	parameters = [
              {'multi_class': ['ovr','crammer_singer'], 'C': [1, 10, 100, 1000]}]

	#load the logistic regression model
	#svm=LinearSVC()

	print("# Tuning hyper-parameters")
	print()

	clf = GridSearchCV(LinearSVC(), parameters , cv=5)
	clf.fit(dtm_train_x,train_dataset['sentiment'].astype(int))

	#svm.fit(dtm_train_x,train_dataset['sentiment'].astype(int))

	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)
	print()
	print("Grid scores on training set:")
	print()
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        	print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
	print()

	#evaluating the model
	train_accuracy=metrics.accuracy_score(train_dataset['sentiment'].astype(int),clf.predict(dtm_train_x))
	pred_y=clf.predict(dtm_test_x)
	test_accuracy=metrics.accuracy_score(test_dataset['sentiment'].astype(int),pred_y)

	#calculate confusion matrix
	cnf_matrix=confusion_matrix(np.array(test_dataset['sentiment'].astype(int)),np.array(pred_y))
	#save confusion matrix as figure
	draw_confusionmatrix(cnf_matrix,classes,"LSVC_parameter_optimization_confusion matrix")

	#finalizing the vectorizer
	vect_file="vectorizer.sav"
	joblib.dump(vect,vect_file)
	
	#finalizing the model
	filename="finalize_model_SVM.sav"
	joblib.dump(clf,filename)

	print("Detailed classification report:")
	print()
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	print()
	
	true_y=test_dataset['sentiment'].astype(int)
	print(classification_report(true_y, pred_y))
	print()

	print "LR train acc:", train_accuracy
	print "LR test acc:" , test_accuracy

if __name__=="__main__":
	main()





