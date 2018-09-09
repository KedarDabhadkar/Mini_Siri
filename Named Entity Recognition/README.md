# Named_Entity_Recognition

![NER](http://174.138.108.232/wp-content/uploads/2017/08/entities.png)<br>

A huge dataset of about 50,000 sentences has been analyzed for making a Named Entity Recognition product. This length is about 10x as much as the length of a mid-size book. The final aim is to assign tags to each and every word. The dataset has been obtained from this Kaggle database. We selectively use the first column, which gives the individual words and the third column, which gives the corresponding tags for our feature engineering experiments.

NER is done using logistic regression (LogReg) in two ways-
1. In the first method, the individual words are used as they are,
2. In the second method, sequential moving windows are generated to introduce a continuity.

Files are arranged in the following format-
Model1_custom_LogReg.py: Custom built LogReg using model 1.
Model1_sklearn_LogReg.py: Sklearn implemented LogReg using model 1.
Model2_custom_LogReg.py: Custom built LogReg using model 2.
Model2_sklearn_LogReg.py: Sklearn implemented LogReg using model 2.

The scripts were run on AWS p2.xlarge GPU instances to prevent memory error and compute faster.
Requires python version >3.5 to execute.
Run using the following command to store results:
python script.py > results.txt

WARNING: If run on a local CPU, might cause memory error and might take a very long time to run.
