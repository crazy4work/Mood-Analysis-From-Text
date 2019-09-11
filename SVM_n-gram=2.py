import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

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
	plt.savefig('cm_SVM.png')


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
	#print(len(vect.get_feature_names()))
	dtm_test_x=vect.transform(test_dataset['content'])

	#load the logistic regression model
	svm=SVC(kernel='linear',decision_function_shape='ovr')
	svm.fit(dtm_train_x,train_dataset['sentiment'].astype(int))

	#evaluating the model
	train_accuracy=metrics.accuracy_score(train_dataset['sentiment'].astype(int),svm.predict(dtm_train_x))
	pred_y=svm.predict(dtm_test_x)
	test_accuracy=metrics.accuracy_score(test_dataset['sentiment'].astype(int),pred_y)

	#calculate confusion matrix
	cnf_matrix=confusion_matrix(np.array(test_dataset['sentiment'].astype(int)),np.array(pred_y))
	#save confusion matrix as figure
	draw_confusionmatrix(cnf_matrix,classes,"SVC_n-gram=2_confusion matrix")

	#finalizing the vectorizer
	vect_file="vectorizer.sav"
	joblib.dump(vect,vect_file)
	
	#finalizing the model
	filename="finalize_model_SVM.sav"
	joblib.dump(svm,filename)

	print "LR train acc:", train_accuracy
	print "LR test acc:" , test_accuracy

if __name__=="__main__":
	main()





