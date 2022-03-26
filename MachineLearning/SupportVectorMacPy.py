#import Libraries
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split #Import train_test function
from sklearn import metrics #Import metrics module for accuracy cal

dataset=pd.read_csv(r"C:\Users\nazsnmz\Desktop\Veri\heart.csv", sep=",") #Load dataset

"""
print(dataset.head()) #Shows firstly 5 rows
print(dataset.info()) #gives information about dataset

print(dataset.columns)
"""
#Split Dataset; Features-Target
#X = dataset.iloc[:, 0:12] #features
#Y = dataset.iloc[:, 13] #target
feature_cols=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']
X=dataset[feature_cols] #features
Y=dataset.target #target

#Train-Test data(%75-%25)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

SupVec=svm.SVC(kernel="linear")
SupVec.fit(X_train, Y_train)
pred=SupVec.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test,pred))