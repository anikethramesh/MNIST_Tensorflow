import tensorflow as tf
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:

	def __init__(self,input_dimensions, output_dimensions, no_hiddenLayers, neurons_hidden_layers_list):
		self.X = tf.placeholder("float", [None,input_dimensions])
		self.Y = tf.placeholder("float", [None,output_dimensions])
		self.dropout_placeholder = tf.placeholder(tf.float32)
		self.number_of_hidden_layers = no_hiddenLayers
		self.NumberOfNeurons_EachLayer = neurons_hidden_layers_list
		self.NumberOfNeurons_EachLayer.insert(0,input_dimensions)
		self.NumberOfNeurons_EachLayer.append(output_dimensions)

	def inputPlaceholder(self):
		return self.X

	def outputPlaceholder(self):
		return self.Y

	def dropoutPlaceholder(self):
		return self.dropout_placeholder

	def createWeightsVector(self):
		self.WeightsKeys = [('w_'+str(i)) for i in range(self.number_of_hidden_layers+1)]
		self.WeightsKeys[-1] = 'w_out'
		self.weights = dict.fromkeys(self.WeightsKeys,)
		for i in range(len(self.WeightsKeys)):
			self.weights[self.WeightsKeys[i]] = tf.Variable(tf.truncated_normal([self.NumberOfNeurons_EachLayer[i],self.NumberOfNeurons_EachLayer[i+1]], stddev = 0.1))
		return self.weights

	def createBiasVector(self):
		self.BiasKeys = [('b_'+str(i)) for i in range(self.number_of_hidden_layers+1)]
		self.BiasKeys[-1] = 'b_out'
		self.bias = dict.fromkeys(self.BiasKeys,)
		for i in range(len(self.BiasKeys)):
			self.bias[self.BiasKeys[i]] = tf.Variable(tf.constant(0.1, shape = [self.NumberOfNeurons_EachLayer[i+1]]))
		return self.bias

	def createLayers(self):
		self.LayerKeys = [('layer_'+str(i)) for i in range(self.number_of_hidden_layers+1)]
		self.LayerKeys[-1] = 'layer_out'
		self.Layers = dict.fromkeys(self.LayerKeys,)
		for i in range(len(self.LayerKeys)):
			if(i==0):
				self.Layers[self.LayerKeys[i]] = tf.add(tf.matmul(self.X,self.weights[self.WeightsKeys[i]]),self.bias[self.BiasKeys[i]])
			else:
				self.Layers[self.LayerKeys[i]] = tf.add(tf.matmul(self.Layers[self.LayerKeys[i-1]],self.weights[self.WeightsKeys[i]]),self.bias[self.BiasKeys[i]])
		return self.Layers

	def createDropoutLayer(self,discard_rate):
		self.dropout_layer = tf.nn.dropout(self.Layers[self.LayerKeys[-2]],discard_rate)
		return self.dropout_layer

 
class TrainNetwork:

	def __init__(self,learningRate,useDropout,num_iter):
		self.learningRate = learningRate
		self.useDropout = useDropout
		self.numberOfIterations = num_iter
		self.minibatch_accuracy_list = []
		self.output_accuracy_list = []
		self.minibatch_loss_list = []
		self.output_loss_list = []

	# def get_Session(self):
	# 	init = tf.global_variables_initializer()
	# 	with tf.Session() as sess:
	# 		sess.run(init)
	# 		return sess

	def set_trainingData(self, X_train,y_train):
		self.X_train = X_train
		self.y_train = y_train

	def set_dropout(self,dropout_val):
		self.dropout = dropout_val

	def set_validationData(self, X_val,y_val):
		self.X_val = X_val
		self.y_val = y_val

	def set_testData(self, X_test,y_test):
		self.X_test = X_test
		self.y_test = y_test

	def set_placeholders(self, X_placeholder, y_placeholder, keep_prob_placeholder):
		self.X = X_placeholder
		self.Y = y_placeholder
		self.keep_prob = keep_prob_placeholder

	def get_lossTensor(self, y_placeholder,output_layer):
		cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_placeholder, logits= output_layer))
		return cross_entropy

	def get_optimizerTensor(self, learningRate, lossTensor):
		train_step = tf.train.AdamOptimizer(learningRate).minimize(lossTensor)
		return train_step

	def get_accuracyTensor(self, y_placeholder,output_layer):
		correct_pred = tf.equal(tf.argmax(output_layer,1),tf.argmax(y_placeholder,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		return self.accuracy

	def startTraining(self,session,lossTensor,optimizerTensor,accuracyTensor):
		for i in range(self.numberOfIterations):
			session.run(optimizerTensor,feed_dict={
				self.X : self.X_train,
				self.Y : self.y_train,
				self.keep_prob : self.dropout
				})

			minibatch_loss, minibatch_accuracy = session.run([optimizerTensor, accuracyTensor],feed_dict = {
				self.X: self.X_val, 
				self.Y: self.y_val, 
				self.keep_prob : 1.0
				})
			self.minibatch_accuracy_list.append(minibatch_accuracy)
			self.minibatch_loss_list.append(minibatch_loss)

			output_loss, output_accuracy = session.run([optimizerTensor, accuracyTensor],feed_dict = {
				self.X: self.X_test, 
				self.Y: self.y_test, 
				self.keep_prob : 1.0 
				})
			self.output_accuracy_list.append(output_accuracy)
			self.output_loss_list.append(output_loss)
			print("Iteration", str(i),"\t|Validation Loss = ", str(minibatch_loss),"\t|Validation Accuracy =", str(minibatch_accuracy),"\nIteration", str(i),"\t|Test Loss = ", str(output_loss),"\t|Test Accuracy =", str(output_accuracy))
			