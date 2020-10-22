#formula of linear regression > y = wx+b > gradient of the slope (W) and bias (b) through multiple iterations.
#learning rate 0.1 has more impact then other
import numpy as np
import tensorflow as tf
import pandas as pd


#setting the datas and showing in a clear form
data = pd.read_csv('Dataset/Admission_Predict.csv')
#print("Data Shape:", data.shape) 
#print(data.head())



# Feature Matrix 
x_orig = data.iloc[:,:-1].values
print("x_orig : ",x_orig)
x_orig = x_orig[:]
#print(x_orig)


# Data labels 
y_orig = data.iloc[:, -1:].values
#print y_orig
y_orig = y_orig[:]
#print(y_orig)




x = x_orig
#print(x)
y = y_orig
#print(x,y)
_,m = x.shape
#print(_)
_,n = y.shape
#print(_)
print(m,n)
print x


#hyperparameters
#learning_rate = 0.01
#slope * x + noise

X = tf.placeholder(tf.float64, shape=[None,m], name="Input")
Y = tf.placeholder(tf.float64, shape=[None,n], name="Output")


W = tf.Variable(np.random.randn(m,n)*np.sqrt(1/(m+n)),name = "weights")
#W = tf.Variable(W*np.sqrt(2/(m+n)))

B = tf.Variable(np.random.randn(n),name = "bias")
#B = tf.Variable(B*np.sqrt(2/(m+n)))

pred =tf.add(tf.matmul(X,W),B)

cost = tf.sqrt(tf.reduce_mean(tf.square(pred - Y))) #tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(pred), reduction_indices=[1]))
# #formula 1/2nE(pred(i)-Y(i))^2 = loss

optimizer = tf.compat.v1.train.AdamOptimizer(0.0001).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
epoch = 0
c = 0
while(epoch<6000):
	epoch+=1
	sess.run(optimizer,feed_dict = {X:x,Y:y})
	if not epoch%1:
		
		c = sess.run(cost,feed_dict = {X:x,Y:y})
		w1 = sess.run(W)
		b = sess.run(B)
		
		if round(c,2) == 0.0:
			break
		print("epoch = ",epoch,"cost = ",c,"Weights1 = ",w1,"Bias = ",b)
