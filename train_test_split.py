import sys
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split

data_path=sys.argv[1]

data=pd.read_csv(data_path,usecols=['sentiment','content'])
train_x,test_x,train_y,test_y=train_test_split(data['content'],data['sentiment'],train_size=0.9)

trf=pd.DataFrame({"content":np.array(train_x),"sentiment":np.array(train_y)})
trf.to_csv("train.csv", index=False)

tef=pd.DataFrame({"content":np.array(test_x), "sentiment":np.array(test_y)})
tef.to_csv("test.csv", index=False)


