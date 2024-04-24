import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


sonar_data = pd.read_csv('sonar.all-data.csv',header = None)
print("Importing data ...")
x = sonar_data.drop(columns = 60, axis = 1)
y = sonar_data[60]

print("Spliting data into smaller dataset for training/testing ...")
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, stratify=y, random_state=1)

print("Model Trainin -->Logistic Regression")
model = LogisticRegression()
model.fit(x_train,y_train)


x_train_predict = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predict, y_train)
print('\tAccuracy on trainin data:' , training_data_accuracy)


x_test_predict = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predict, y_test)
print('\tAccuracy on test data:' , test_data_accuracy)


input_data = (0.0310,0.0221,0.0433,0.0191,0.0964,0.1827,0.1106,0.1702,0.2804,0.4432,0.5222,0.5611,0.5379,0.4048,0.2245,0.1784,0.2297,0.2720,0.5209,0.6898,0.8202,0.8780,0.7600,0.7616,0.7152,0.7288,0.8686,0.9509,0.8348,0.5730,0.4363,0.4289,0.4240,0.3156,0.1287,0.1477,0.2062,0.2400,0.5173,0.5168,0.1491,0.2407,0.3415,0.4494,0.4624,0.2001,0.0775,0.1232,0.0783,0.0089,0.0249,0.0204,0.0059,0.0053,0.0079,0.0037,0.0015,0.0056,0.0067,0.0054)
#changing the input data
numpy_array = np.asarray(input_data)
input_data_reshape = numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshape)

if prediction[0] == 'M':
  print("The Object is a Mine")
else:
    print("The Object is a rock")