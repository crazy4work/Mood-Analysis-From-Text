import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import itertools
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.externals import joblib


DATAPATH="/home/suparna/Mood Analysis/Text/amazon/text_emotion.csv"

DATA_REALIZATION=open("data_realization.txt",'a')
def words_in_each_class(Y,dict_label,vocabulary,j_indices,indptr,value):
	#print Y[0]
	#print Y[1]
	#print Y[2]
	#print Y[3]
	#print Y[4]
	#print Y[5]
	#print Y[6]
	d={}
	l=len(Y.index)
	print l
	#classes=dict_label.keys()
	#class_score=dict_label.values()
	words=[x.encode('ascii','ignore') for x in vocabulary.keys()]
	print len(words)
	#print words
	indices=vocabulary.values()
	print len(indices)
	for i in Y:
		#print i
		if not i in d.keys():
			d[i]={}
		for j in range(indptr[i],indptr[i+1]):
			v=j_indices[j]
			print v
			n=indices.index(v)
			print n
			key=words[n]
			print key
			if not key in d[i].keys():
				d[i][key]=value[j]
			else:
				d[i][key]+=value[j]
	#print d

	DATA_REALIZATION.write("Vocabulary={}\n\n".format(vocabulary))
	DATA_REALIZATION.write("words in each class: {}\n\n".format(d))
	for comb in itertools.combinations(d,2):
		DATA_REALIZATION.write("{} : {}\n\n".format(comb,(d[comb[0]].viewkeys() & d[comb[1]].viewkeys())))

def count_instances_per_class(Y):
	instances_per_class=Counter(Y)
	l=len(Y.index)
	DATA_REALIZATION.write("instances per class:{}  {}\n\n".format(instances_per_class,l))

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
	plt.xticks(tick_mark,classes,rotation=45)
	#marking y axis
	plt.yticks(tick_mark,classes)
	#writing the confusion matrix
	for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
		plt.text(j,i,format(cm[i][j],'d'),horizontalalignment='center')
	plt.ylabel('true label')
	plt.xlabel('predicted label')
	plt.savefig('cm_lr.png')

def main():
	
	#load dataset
	dataset=pd.read_csv(DATAPATH,usecols=['sentiment','content'])
	#Y=dataset['sentiment']
	#print dataset['sentiment']

	#encoding lables
	Y=dataset['sentiment']
	#print Y
	#print type(dataset["sentiment"])
	#print type(Y)
	set_Y=set(Y)
	#print set_Y
	dict_label={}
	i=0
	for x in set_Y:
		dict_label[x]=i
		i+=1

	print dict_label

	
	DATA_REALIZATION.write("class coding: {}\n\n".format(dict_label))

	for i in xrange(len(Y)):
		Y[i]=dict_label[Y[i]]
	#print type(Y)
	count_instances_per_class(Y)
	#train test split
	train_x,test_x,train_y,test_y=train_test_split(dataset['content'],Y,train_size=0.9)

	count_instances_per_class(train_y)


	#feature extraction from text : CountVectorizer
	vect=CountVectorizer()
	dtm_train_x,j_indices,indptr,value=vect.fit_transform(train_x)
	#print dtm_train_x
	DATA_REALIZATION.write("J_INDices={}, size={}\n".format(j_indices,len(j_indices)))
	DATA_REALIZATION.write("indptr={}, size={}\n".format(indptr,len(indptr)))
	DATA_REALIZATION.write("value={}, size={}\n".format(value,len(value)))
	#data visualization
	vocabulary=vect.vocabulary_
	words_in_each_class(train_y.astype(int),dict_label,vocabulary,j_indices,indptr,value)


	dtm_test_x=vect.transform(test_x)

	#load the logistic regression model
	mul_log_reg=linear_model.LogisticRegression(multi_class="multinomial",solver="newton-cg")
	mul_log_reg.fit(dtm_train_x,train_y.astype(int))

	#evaluating the model
	train_accuracy=metrics.accuracy_score(train_y.astype(int),mul_log_reg.predict(dtm_train_x))
	pred_y=mul_log_reg.predict(dtm_test_x)
	test_accuracy=metrics.accuracy_score(test_y.astype(int),pred_y)

	#calculate confusion matrix
	cnf_matrix=confusion_matrix(np.array(test_y.astype(int)),np.array(pred_y))
	#save confusion matrix as figure
	draw_confusionmatrix(cnf_matrix,set_Y,"confusion matrix")

	#finalizing the vectorizer
	vect_file="vectorizer.sav"
	joblib.dump(vect,vect_file)
	
	#finalizing the model
	filename="finalize_model_LR.sav"
	joblib.dump(mul_log_reg,filename)

	print "LR train acc:", train_accuracy
	print "LR test acc:" , test_accuracy

if __name__=="__main__":
	main()
