
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

file ='ner_dataset.csv'
doc = open(file,'r')

st = doc.read(100000)

rel = st.splitlines()

max_epoch = 50

map_={}
list_=[]
for i in rel[1:]:
    text = i.split(sep=',')
    if len(text) == 4:
        map_[text[1]] = text[3]
        list_.append([text[1],text[3]])

all_data = np.char.lower(list_)
train = all_data[0:int(0.8*len(list_))]
valid = all_data[int(0.8*len(list_)):int(0.9*len(list_))]
test = all_data[int(0.9*len(list_)):int(len(list_))]

def Indi (a , b):
    if a == b :
        return 1
    else :
        return 0

# Making feature matrices for trainng, testing and validation
X=np.char.lower(train[:,0])
Xvalid=np.char.lower(valid[:,0])
Xtest=np.char.lower(test[:,0])

# Generate class vectors
class_=train[:,-1]
class_valid=valid[:,-1]
class_test=test[:,-1]

#Unique train labels
uniq_class=np.unique(all_data[:,-1])

#Get unique words
words=np.unique(all_data[:,0])

#Define numbers
M=len(words) + 1 #Number of rows of feature matrix for training
K=len(uniq_class) #Number of columns of feature matrix for training

N=len(class_) #Length of class vector for training
Ntest=len(Xtest) #Length of class vector for testing
Nvalid=len(Xvalid) #Length of class vector for validation

Xttrans=[]
class_num=[]
for i,example in enumerate(X):
    Xttrans.append(words.tolist().index(example))
    class_num.append(uniq_class.tolist().index(class_[i]))

Xvtrans=[]
classv_num=[]
for i,example in enumerate(Xvalid):
    Xvtrans.append(words.tolist().index(example))
    classv_num.append(uniq_class.tolist().index(class_valid[i]))

Xtesttrans=[]
classtest_num=[]
for i,example in enumerate(Xtest):
    Xtesttrans.append(words.tolist().index(example))
    classtest_num.append(uniq_class.tolist().index(class_test[i]))
    
theta_init=np.matrix(np.zeros([M,K]))


# In[2]:


from sklearn.linear_model import LogisticRegression


# In[10]:


log = LogisticRegression(penalty='l1',class_weight='balanced',verbose=1)
log.fit(np.array(Xttrans).reshape(-1,1),class_)
predicted = log.predict(np.array(Xtesttrans).reshape(-1,1))


# In[11]:


precision, recall, fscore, class_count = score(class_test, predicted, labels=uniq_class)
print ('\n Classification metrics: \n')
for i in range(uniq_class.shape[0]):
    print (uniq_class[i])
    print ('\tprecision: ' + '{0:.5f}'.format(precision[i]))
    print ('\trecall: ' + '{0:.5f}'.format(recall[i]))
    print ('\tfscore: ' + '{0:.5f}'.format(fscore[i]))
    print ('\tTotal datapoints under this class: {}'.format(class_count[i]))

print ('\nF1 MACRO: {0:.5f}'.format(np.mean(fscore)))


# In[12]:


Xttrans

