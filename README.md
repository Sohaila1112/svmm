from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('/content/Cancer_Data.csv')





X = df.drop(['id', 'diagnosis', 'Unnamed: 32'] ,   axis = 1)
X



le = LabelEncoder()
Y = le.fit_transform(Y)



X_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)




svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)




svm.fit(X_train, y_train)



y_pred = svm.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Precision:", precision_score(Y_test, y_pred, average='macro'))
print("Recall:", recall_score(Y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred, average='macro'))

