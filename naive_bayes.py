

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

Data = pd.read_csv("/content/Wholedata.csv")

Data.head()

Data.dropna(inplace=True)

Data.head()

Data = Data.drop(["Date"], axis=1)

TrainDF, TestDF = train_test_split(Data, test_size=0.30)

TestDF.head()

TrainLabels=TrainDF["Campaign Name"]

TrainDF = TrainDF.drop(["Campaign Name"], axis=1)

TestLabels=TestDF["Campaign Name"]

TestDF = TestDF.drop(["Campaign Name"], axis=1)

"""The dataset was divided into two subsets: a training dataset comprising 70% of the data and a testing dataset comprising the remaining 30%. This partitioning ensures that the model is trained on one set of data and tested on another, disjoint set, preventing any potential bias in the evaluation process. The model was trained using the training dataset, and its performance was assessed using the testing dataset. This approach helps to provide an unbiased evaluation of the model's performance on unseen data.


"""

TrainDF.head()

TestDF.head()



MyModelNB1= MultinomialNB()

MyModelNB1.fit(TrainDF, TrainLabels)

Prediction1 = MyModelNB1.predict(TestDF)

Prediction1

Proability  = np.round(MyModelNB1.predict_proba(TestDF),2)

Proability

TrainCamp = Proability[:,0]

TestCamp = Proability[:,1]

from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import ConfusionMatrixDisplay

cnf_matrix1 = confusion_matrix(TestLabels, Prediction1)

cnf_matrix1

disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix1,
                               display_labels=MyModelNB1.classes_)

disp.plot()

from sklearn.metrics import f1_score

print(classification_report(TestLabels, Prediction1, target_names=MyModelNB1.classes_))

from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import ConfusionMatrixDisplay
Prediction1 = MyDT.predict(TestDF)

cnf_matrix1 = confusion_matrix(TestLabels, Prediction1)
cnf_matrix1

disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix1,
                               display_labels=MyDT.classes_)

plt.figure(figsize=(100,30))
disp.plot()

print(classification_report(TestLabels, Prediction1, target_names=MyDT.classes_))


