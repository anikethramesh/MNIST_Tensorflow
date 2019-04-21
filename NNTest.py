import pandas as pd
import tensorflow as tf
import scipy
import matplotlib.pyplot as plt
from ML_Pipeline import SimpleNeuralNetwork, TrainNetwork

df_train = pd.read_csv('MNIST_data/mnist_train.csv',sep=',')
target = df_train['label'].astype(float)
train = df_train.drop("label",axis=1).astype(float)

#Test Data
df_test = pd.read_csv('MNIST_data/mnist_test.csv',sep=',')
test_output = df_test['label'].astype(float)
test_input = df_test.drop("label",axis=1).astype(float)

b = train.var()
train = train.drop(b[b[:]<2000].index,axis = 1)
test_input = test_input.drop(b[b[:]<2000].index,axis=1)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train.to_numpy(),target.to_numpy(),test_size = 0.15, random_state=0)

test_input = test_input.to_numpy()
test_output = test_output.to_numpy()

#Scaler
from sklearn.preprocessing import StandardScaler
train_scaler = StandardScaler()
validation_scaler = StandardScaler()
test_scaler = StandardScaler()

X_train = train_scaler.fit_transform(X_train)
X_val = train_scaler.fit_transform(X_val) 
test_input = test_scaler.fit_transform(test_input)
##
y_train = y_train.reshape(-1,1)
y_val = y_val.reshape(-1,1)
test_output = test_output.reshape(-1,1)

#Encoding output
from sklearn.preprocessing import OneHotEncoder
ohe_train = OneHotEncoder(categories='auto')
ohe_val = OneHotEncoder(categories='auto')
ohe_test = OneHotEncoder(categories = 'auto')
y_train = ohe_train.fit_transform(y_train).toarray()
y_val = ohe_val.fit_transform(y_val).toarray()
test_output = ohe_test.fit_transform(test_output).toarray()

[m_examples,n_features] = X_train.shape

[m_test , n_test] = test_input.shape

######Machine Learning Part######

#Create Neural Network
network = SimpleNeuralNetwork(n_features,10,3,[256,128,64])
X = network.inputPlaceholder()
Y = network.outputPlaceholder()
dropout = network.dropoutPlaceholder()
weights = network.createWeightsVector()
bias = network.createBiasVector()
layers  = network.createLayers()
dropoutLayer = network.createDropoutLayer(0.5)

#Setup Training
trainer = TrainNetwork(1e-3,True,1000)
trainer.set_trainingData(X_train,y_train)
trainer.set_validationData(X_val,y_val)
trainer.set_testData(test_input,test_output)
trainer.set_dropout(0.5)
trainer.set_placeholders(X,Y,dropout)
loss = trainer.get_lossTensor(Y,layers['layer_out'])
optimizer = trainer.get_optimizerTensor(1e-3,loss)
accuracy = trainer.get_accuracyTensor(Y,layers['layer_out'])
init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)
	trainer.startTraining(session,loss,optimizer,accuracy)