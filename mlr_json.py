import json
from sklearn.linear_model import LogisticRegression

class MyMLR(LogisticRegression):
	''' this class is to dump LogisticRegression for multi class from sklearn as JSON file manually; Here we will implement save_json and 	load_json method'''

	# override the class constructor
	def __init__(self,c=1.0,solver="newton-cg",multi_class="multimonial",max_iter=100,X_train=none,Y_train=None):
		LogisticRegression.__init__(self,c=c,solver=solver,multi_class=multi_class,max_iter=max_iter)
		self.X_train=X_train
		self.Y_train=Y_train

	#a method for saving object data to file
	def save_json(self, dict, filepath):
		dic={}
		dic['coef_']=self.coef_
		dic['multi_class']=self.multiclass
		json_text=json.dump(dic)
		with open(filepath,'w+') as file:
			file.write(json_text)

	# method for loading data from json file
	def load_json(self, filepath):
		with open(filepath,'r') as file:
			dic=json.load(file)
		self.coef_=dic['coef_']
		self.multi_class=dic['multi_class']
