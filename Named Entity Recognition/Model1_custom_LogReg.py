# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:30:26 2018

@author: kdabhadk
"""

import time

# Start timer
start = time.time()

# import required modules
import numpy as np
import codecs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

# Import data
file ='ner_dataset.csv'
with codecs.open(file, "r",encoding='utf-8', errors='ignore') as doc:
    st = doc.read()

# Set maximum number of epochs
max_epoch = 5

#Split sccording to lines
rel = st.splitlines()

#Selectively parse the first and third columns and store them in a list
list_=[]
for i in rel[1:]:
    text = i.split(sep=',')
    if len(text) == 4:
        list_.append([text[1],text[3]])

#Store all the data in the numpy object all_data
#Cover all characters to lower case
all_data = np.char.lower(list_)

#Split training as the first 80% of all data
train = all_data[0:int(0.8*len(list_))]

# validation fraction is 10%
valid = all_data[int(0.8*len(list_)):int(0.9*len(list_))]

# Test fraction is 10%
test = all_data[int(0.9*len(list_)):int(len(list_))]

# Define the indicator function required in training
def Indi (a , b):
    '''
    Returns 1 if a=b, else returns 0.

    Parametes
    ----------
    a,b : Objects

    Returns
    ----------
    Boolean, depending on similarity of objects a and b.
    '''
    if a == b :
        return 1
    else :
        return 0


# Making feature matrices for trainng, testing and validation
X=np.char.lower(train[:,0]) #Feature training matrix
Xvalid=np.char.lower(valid[:,0]) #Feature validation matrix
Xtest=np.char.lower(test[:,0]) #Feature test matrix

# Generate class vectors
class_=train[:,-1] #Training labels
class_valid=valid[:,-1] #Validation labels
class_test=test[:,-1] #Test labels

#Unique train labels
uniq_class = np.unique(all_data[:,-1]) #Array of all unique labels

#Get unique words
words = np.unique(all_data[:,0]) #Array of all unique words

#Define numbers
M=len(words) + 1 #Number of rows of feature matrix for training
K=len(uniq_class) #Number of columns of feature matrix for training

N=len(class_) #Length of class vector for training
Ntest=len(Xtest) #Length of class vector for testing
Nvalid=len(Xvalid) #Length of class vector for validation

# Numerical array for training
Xttrans=[]
class_num=[]
for i,example in enumerate(X):
    Xttrans.append(words.tolist().index(example))
    class_num.append(uniq_class.tolist().index(class_[i]))

# Numerical array for valiation
Xvtrans=[]
classv_num=[]
for i,example in enumerate(Xvalid):
    Xvtrans.append(words.tolist().index(example))
    classv_num.append(uniq_class.tolist().index(class_valid[i]))

# Numerical array for test
Xtesttrans=[]
classtest_num=[]
for i,example in enumerate(Xtest):
    Xtesttrans.append(words.tolist().index(example))
    classtest_num.append(uniq_class.tolist().index(class_test[i]))

#Initialize a parameter matrix
theta_init=np.matrix(np.zeros([M,K]))

epoch=0

#Define a Stochastic Gradient Descent function for training
#Train theta using SGD and print NLL
def SGD(Xttrans,class_,theta_init,max_epoch):
    '''
    Stochastic Gradient Descent (SGD) for training parameter matrix
    using negative log likelihood as the error metric.

    Parameters:
    ----------
    1. Xttrans: (np.matrix) Numerical feature matrix for training.
    2. class_: (list) List of class labels of the training data.
    3. theta_init: (np.matrix) Initialization of parameter matrix.
    4. max_epoch: (int) Number of epochs.

    Returns:
    ----------
    1. theta: (np.matrix) Trained parameter matrix.
    2. grad: (np.matrix) Matrix of gradient of negative log likelihood with
                respect to the parameter matrix theta.
    3. nlltrain: (list) List of negative log likelihood on the training data after every epoch.
    4. nllvalid: (list) List of negative log likelihood on the validation data after every epoch.
    '''

    # Initialize all result parameters
    theta=theta_init
    nllvalid=[]
    nlltrain=[]
    Nvalid=len(Xvtrans)

    # Loop to perform SGD.
    epoch = 0
    while epoch < max_epoch:
        nll=0
        grad = np.matrix(np.empty([M,K]))

        for i,example in enumerate(Xttrans): # Loop over N
            denom=0
            num=0

            for z in range(K):
                denom=denom+(np.exp(theta[example,z]+theta[-1,z]))

            for j,k in enumerate(range(K)): # Loop over K
                num=np.exp(theta[example,k]+theta[-1,k])

                I=Indi(class_[i],uniq_class[k])

                ans=-(I - num/denom)
                grad[example,k] = ans
                grad[-1,k] = ans

            # Update rule
            theta[example] = theta[example] - 0.5 * grad[example]

            theta[-1] = theta[-1] - 0.5 * grad[-1]
            theta_init=theta

        # Keep track of negative log likelihood on training data
        nll=0
        for i,example in enumerate(Xttrans):
            denom=0
            num=0
            for z in range(K):
                denom=denom+(np.exp(theta[example,z]+theta[-1,z]))

            for j,k in enumerate(range(K)):
                num=np.exp(theta[example,k]+theta[-1,k])
                nll=nll-Indi(class_[i],uniq_class[k])/N*np.log(num/denom)
        nlltrain.append(nll)

        # Keep track of negative log likelihood on validation data
        nll=0
        for i,example in enumerate(Xvtrans):
            denom=0
            num=0
            for z in range(K):
                denom=denom+(np.exp(theta[example,z]+theta[-1,z]))

            for j,k in enumerate(range(K)):
                num=np.exp(theta[example,k]+theta[-1,k])
                nll=nll-Indi(class_valid[i],uniq_class[k])/Nvalid*np.log(num/denom)
        nllvalid.append(nll)
        epoch = epoch + 1
        print ('Completed epoch: {}'.format(epoch))
    return theta,grad,nlltrain,nllvalid

theta,grad,nlltrain,nllvalid=SGD(Xttrans,class_,theta_init,max_epoch)

# Test predictions
predtest=[]
errtest = 0
for i,example in enumerate(Xtesttrans):
    num=[]
    for j,k in enumerate(range(K)): #Over K
        num.append(theta[example,k]+theta[-1,k])
    predtest.append(uniq_class[np.argmax(num)])
    if predtest[i] != class_test[i]:
        errtest= errtest + 1

# Train predictions
predtrain=[]
errtrain = 0
for i,example in enumerate(Xttrans):
    num=[]
    for j,k in enumerate(range(K)): #Over K
        num.append(theta[example,k]+theta[-1,k])
    predtrain.append(uniq_class[np.argmax(num)])
    if predtrain[i] != class_[i]:
        errtrain= errtrain + 1

# Errors on the training and test data
error_train = np.round(errtrain/N,decimals=6)
error_test = np.round(errtest/Ntest,decimals=6)

# Calculate other error metrics
print ('Confusion matrix:')
print(confusion_matrix(class_test,predtest))

print ('F1-Scores:')
print(score(class_test,predtest)[2])

precision, recall, fscore, class_count = score(class_test, predtest, labels=uniq_class)
print ('\n Classification metrics: \n')

for i in range(uniq_class.shape[0]):
    print (uniq_class[i])
    print ('\tprecision: ' + '{0:.5f}'.format(precision[i]))
    print ('\trecall: ' + '{0:.5f}'.format(recall[i]))
    print ('\tfscore: ' + '{0:.5f}'.format(fscore[i]))
    print ('\tTotal datapoints under this class: {}'.format(class_count[i]))

print ('\nAverage F1 MACRO: {0:.5f}'.format(np.sum(np.array(class_count)*np.array(fscore))/sum(class_count)))

end = time.time()

print ('Total run time = {}'.format(end-start))
