#Import Libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier #Import Decision Tree Classifier
from sklearn.model_selection import train_test_split #Import train_test function
from sklearn import metrics #Import metrics module for accuracy cal


dataset=pd.read_csv(r"C:\Users\nazsnmz\Desktop\Veri\heart.csv", sep=",") #Load Dataset
"""
print(dataset.head()) #Shows firstly 5 rows
print(dataset.info()) #gives information about dataset

print(dataset.columns)
"""
#split dataset in features and target variable

feature_cols=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']
X=dataset[feature_cols] #features
Y=dataset.target #target
"""
X=dataset.iloc[:,0:12] #features
Y=dataset.iloc[:,13] #target
"""
#Split dataset Train-Test (%70 train-%30 test)
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.3, random_state=1)

#Create Decision Tree Model
DesTree=DecisionTreeClassifier()
DesTree=DesTree.fit(X_train,Y_train) #Train Decision Tree
pred=DesTree.predict(X_test) #predict the response for test dataset

print("Accuracy:", metrics.accuracy_score(Y_test, pred)) #Doğruluk :0.99..