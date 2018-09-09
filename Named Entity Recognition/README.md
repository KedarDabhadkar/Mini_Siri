# Named_Entity_Recognition

![NER](http://174.138.108.232/wp-content/uploads/2017/08/entities.png)<br>

A huge dataset of about 50,000 sentences has been analyzed for making a Named Entity Recognition product. This length is about 10x as much as the length of a mid-size book. The final aim is to assign tags to each and every word. The dataset has been obtained from this Kaggle database. We selectively use the first column, which gives the individual words and the third column, which gives the corresponding tags for our feature engineering experiments.<br>

NER is done using logistic regression (LogReg) in two ways-<br>
1. In the first method, the individual words are used as they are,<br>
2. In the second method, sequential moving windows are generated to introduce a continuity.<br>

Files are arranged in the following format- <br>
Model1_custom_LogReg.py: Custom built LogReg using model 1.<br>
Model1_sklearn_LogReg.py: Sklearn implemented LogReg using model 1.<br>
Model2_custom_LogReg.py: Custom built LogReg using model 2.<br>
Model2_sklearn_LogReg.py: Sklearn implemented LogReg using model 2.<br>

The scripts were run on AWS p2.xlarge GPU instances to prevent memory error and compute faster.<br>
Requires python version >3.5 to execute.<br>
Run using the following command to store results:<br>
python script.py > results.txt<br>

WARNING: If run on a local CPU, might cause memory error and might take a very long time to run.
