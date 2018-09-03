
# coding: utf-8

# In[4]:

import time
start = time.time()

import numpy as np
import pandas as pd
import codecs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.linear_model import LogisticRegression


# In[5]:


file ='ner_dataset.csv'
with codecs.open(file, "r",encoding='utf-8', errors='ignore') as doc:
    st = doc.read(999)

rel = st.splitlines()

max_epoch = 5

list_=[]
for i,datapoint in enumerate(rel[1:]):
    text = datapoint.split(sep=',')
    if len(text) == 4:
        if text[0][0:8]=='Sentence':
            list_.append(['EOS','EOS'])
            list_.append(['BOS','BOS'])
        list_.append([text[1],text[3]])

all_data = np.char.lower(list_)
train = all_data[0:int(0.8*len(list_))]
valid = all_data[int(0.8*len(list_)):int(0.9*len(list_))]
test = all_data[int(0.9*len(list_)):int(len(list_))]

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

M=len(words) + 1
K=len(uniq_class)
N=len(class_)
Ntest=len(Xtest)
Nvalid=len(Xvalid)


# In[6]:


def generate(trainmodify):
    Xttrans=[]
    label=[]

    for i in range(1,len(trainmodify)-1):
        ap=[]
        firstword=trainmodify[i-1,0]
        wordmid=trainmodify[i,0]
        thirdword=trainmodify[i+1,0]

        if wordmid != 'eos':
            if wordmid != 'bos':
                ap.append(words.tolist().index(firstword))
                ap.append(M-1+words.tolist().index(wordmid))
                ap.append(2*M-2+words.tolist().index(thirdword))

                label.append(uniq_class.tolist().index(trainmodify[i,1]))
        if len(ap) != 0:
            Xttrans.append(ap)
    return Xttrans,label


# In[7]:


Xttrans,train_labels=generate(train)
Xvtrans,valid_labels=generate(valid)
Xtesttrans,test_labels=generate(test)


# In[17]:


log = LogisticRegression(penalty='l1',class_weight='balanced')
log.fit(np.array(Xttrans).reshape(-1,3),train_labels)
predicted = log.predict(np.array(Xtesttrans).reshape(-1,3))


# In[19]:

print ('Confusion matrix:')
print(confusion_matrix(test_labels, predicted, labels=np.unique(test_labels)))


# In[54]:
precision, recall, fscore, class_count = score(test_labels, predicted, labels=np.unique(test_labels))

print ('F1-Scores:')
print(fscore)

print ('\n Classification metrics: \n')
for i in range(np.unique(test_labels).shape[0]):
    print (uniq_class[i])
    print ('\tprecision: ' + '{0:.5f}'.format(precision[i]))
    print ('\trecall: ' + '{0:.5f}'.format(recall[i]))
    print ('\tfscore: ' + '{0:.5f}'.format(fscore[i]))
    print ('\tTotal datapoints under this class: {}'.format(class_count[i]))

print ('\nAverage F1 MACRO: {0:.5f}'.format(np.sum(np.array(class_count)*np.array(fscore))/sum(class_count)))

end = time.time()

print ('Total run time:{}'.format(end-start))
