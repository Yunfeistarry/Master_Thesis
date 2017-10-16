
# coding: utf-8

# In[1]:

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
import glob
import tensorflow as tf




# <h1> Choose One out of three different Feature Matrix


#Feature Matrix: 50 features————annual load profile distribution ;Battery capacity; Number of PV modules
X_raw=np.load('/Users/Yunfei/Documents/Master_Thesis/X_raw.npy')
SS_train=np.load('/Users/Yunfei/Documents/Master_Thesis/SS_train.npy')
X2=X_raw[:,0:48]*0.5*365 #change the average daily load profile to annual load profile distribution
X_50_unscaled=np.concatenate((X2,X_raw[:,[48,49]]),1)
print X_50_unscaled.shape #construct the feature matrix

from sklearn.model_selection import train_test_split #split the dataset, 90% for training and validation, 10% for testing
X_training, X_test, Y_training, Y_test = train_test_split(X_50_unscaled,SS_train, test_size=0.1, random_state=42)


print X_training.shape
total_len = X_training.shape[0]
test_len=X_test.shape[0]
print total_len
Y_training=Y_training.reshape(total_len,1)
Y_test=Y_test.reshape(test_len,1)
print Y_training.shape,X_training.shape




# <h1> Neural Network structure

# Parameters
training_epochs =40
batch_size = 500
display_step = 1  #每几个epoch显示一次w,b的结果
batch_size=500
# Network Parameters
n_input = X_training.shape[1]
n_classes = 1


# tensorflow Graph input
x = tf.placeholder("float", [None, 50])
y = tf.placeholder("float", [None,1])

# Create neural network model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Hidden layer with RELU activation

    #Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer



 
n_hidden_1=80
n_hidden_2=40
n_hidden_3=40

weights = {
'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)), #shape, mean, stddev
'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], 0, 0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),      
    'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
}    

# Construct model
pred=multilayer_perceptron(x, weights, biases)

# Define Performance 
cost = tf.reduce_mean(tf.square(pred-y)) #MSE
mean, var = tf.nn.moments((pred-y),axes = [0])


total_error = tf.reduce_sum(tf.square(y-tf.reduce_mean(y)))
unexplained_error = tf.reduce_sum(tf.square(y-pred))
R_squared = 1-tf.div(unexplained_error,total_error)  #R2

#Define Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)







# <h1> Parameter tuning based on Cross Validation
#parameters tuning involving 5-fold cross validation
#Import k-fold cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(X_training)
# print(kf)  
R=np.arange(1)
R_SGD2=np.arange(1)
k=1

with tf.Session() as sess:
    #sess.run(tf.initialize_all_variables()) #tf.global_variables_initializer
    sess.run(tf.global_variables_initializer())
    for train_index, valid_index in kf.split(X_training):
        #Train the model with 5-fold cross validation(using 90% of the whole dataset, so the size of the tranining data set=975052*0.8)
        X_train, X_valid = X_training[train_index], X_training[valid_index]
        Y_train, Y_valid = Y_training[train_index], Y_training[valid_index]
        # Training cycle
       
        for epoch in range(training_epochs): #training_epochs
            avg_cost = 0.
            total_batch = int(X_train.shape[0]/batch_size)
            # Loop over all batches
            for i in range(total_batch-1):
                random_index=np.random.randint(1,high=975052,size=batch_size)
                batch_x=X_train[random_index,:]
                batch_y=Y_train[random_index]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, R2,c, estimate = sess.run([optimizer, R_squared, cost, pred], feed_dict={x: batch_x,y: batch_y})
                # Compute average MSE
                avg_cost += c / total_batch


            R2_valid= sess.run([R_squared], feed_dict={x:X_valid, y: Y_valid})  #想要每跑完一个epoch都在validation set上检验一下R2的值，
            #若超过了0.976则跳出epoch循环，在test set上检验R2的值，若没有,则继续直到完成了25个epoch的更新，然后在test上检验一下R2的值    
            print ("Fold:",k,"Epoch:" ,epoch,"Training Set Average MSE",avg_cost,"VALIDATION set R2:", R2_valid)
            if R2_valid[0]>0.999:
                break
            R_SGD2=np.append(R_SGD2,R2_valid)
            
        k=k+1
        
        testing_cost, R2_test,difmean,difvar,test_output= sess.run([cost,R_squared,mean,var,pred], feed_dict={x:X_test, y: Y_test})
        print("----------------------------------------------------------------------------------------")
        print ("Fold:",k,"TEST Set MSE:" , testing_cost, "TEST set R2:", R2_test,"difmean:", difmean,"difvar:", difvar)
        print("========================================================================================")

print ('R2_Test=%.*f'%(4,RR))






# <h1> Construct different datasets for application
#Construct feature matrix with fixed annual load profile
load=[3,3,2.6,2.6,2.4,2.4,2.4,2.2,2.2,1.8,1.8,1.8,2.2,2.2,2.5,3,3.2,3.4,3.4,3.4,3.4,3.4,3.4,3.4,3.5,3.5,3.5,3.5,3.4,3.4,3.4,3.4,3.5,4,4.5,5,6,7.5,8,7.5,7.5,7.2,6.8,6.4,6,5.6,5.3,5]
load=np.asarray(load)
print load.shape
App2=np.zeros((13,48))
App2.shape
#scale the load profile shape to 5000 kWh annual demand
for i in range(13):
    App2[i,:]=load*5000/np.sum(load)
    
pv=[2,10,20,30,40,50,60,70,80,90,100,110,120]
bat=np.ones(13)*2
pv=np.asarray(pv).reshape(13,1)
bat=np.asarray(bat).reshape(13,1)
App2=np.concatenate((App2,bat),1)
App2=np.concatenate((App2,pv),1)
print App2.shape

#Construct feature matrix with fixed PV
load=[3,3,2.6,2.6,2.4,2.4,2.4,2.2,2.2,1.8,1.8,1.8,2.2,2.2,2.5,3,3.2,3.4,3.4,3.4,3.4,3.4,3.4,3.4,3.5,3.5,3.5,3.5,3.4,3.4,3.4,3.4,3.5,4,4.5,5,6,7.5,8,7.5,7.5,7.2,6.8,6.4,6,5.6,5.3,5]
load=np.asarray(load)
L=np.arange(5,17)
L=L*500
print L[3]
App3=np.zeros((12,48))
print App3.shape

for i in range(12):
    App3[i,:]=load*(L[i]/np.sum(load))
    
pv=np.ones(12)*50.
pv=pv.reshape(12,1)
bat=np.ones(12)*2
bat=bat.reshape(12,1)


App3=np.concatenate((App3,bat),1)
App3=np.concatenate((App3,pv),1)
print App3.shape





# <h1> Train the final model and apply it on the constructed datasets

# In[30]:

#train with the 90% of the whole dataset and test on the rest 10%. Then apply the model to predict the new data(from the clusters)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training cycle
  
    for epoch in range(40): #training_epochs
        avg_cost = 0.
        total_batch = int(X_training.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            random_index=np.random.randint(1,high=975052,size=batch_size)
            batch_x=X_training[random_index,:]
            batch_y=Y_training[random_index]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, R2,c, estimate = sess.run([optimizer, R_squared, cost, pred], feed_dict={x: batch_x,y: batch_y})
            # Compute average MSE
            avg_cost += c / total_batch
            #saver.save(sess=session, save_path=save_path)

  
    testing_cost, R2_test,difmean,difvar,test_output= sess.run([cost,R_squared,mean,var,pred], feed_dict={x:X_test, y: Y_test})
    print("----------------------------------------------------------------------------------------")
    print ("TEST Set MSE:" , testing_cost, "TEST set R2:", R2_test,"difmean:", difmean,"difvar:", difvar)
    print("========================================================================================")
     
    result2= sess.run([pred], feed_dict={x:App2})
    print("----------------------------------------------------------------------------------------")
    print ("APP prediction:" ,result1)
    print("========================================================================================")
    
    
    
    result3= sess.run([pred], feed_dict={x:App3})
    print("----------------------------------------------------------------------------------------")
    print ("APP prediction:" ,result3)
    print("========================================================================================")
    
Result2=np.asarray(result2).reshape(13)
Result3=np.asarray(result3).reshape(12)
np.save('Result2',Result2)
np.save('Result3',Result3)
