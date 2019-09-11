
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

raw_data=["Hey... I am meeting my mother tomorrow and I am very happy","can i meet you tomorrw? It's very urgent."]

#X=pd.DataFrame(raw_data)

#loading extracted feature durin training
features=joblib.load("vectorizer.sav")

#vectorizing test data according to the features
test_x=features.transform(raw_data)

#loading the model ...multinomial logistic regression
model=joblib.load("finalize_model_LR.sav")

#testing
result=model.predict_proba(test_x)
print result
