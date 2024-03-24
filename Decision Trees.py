


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split











Data = pd.read_csv("Wholedata.csv")





Data.head()





Data = Data.drop(["Date"], axis=1)




Data.dropna(inplace=True)





Data.head()





TrainDF, TestDF = train_test_split(Data, test_size=0.30)
TestDF.head()





TrainLabels=TrainDF["Campaign Name"]





TrainDF = TrainDF.drop(["Campaign Name"], axis=1)





TestLabels=TestDF["Campaign Name"]





TestDF = TestDF.drop(["Campaign Name"], axis=1)





TrainDF.head()





TestDF.head()





Features = TestDF.columns.tolist()





MyDT=DecisionTreeClassifier(criterion='entropy', 
                            splitter='best')





t = MyDT.fit(TrainDF, TrainLabels)

plt.figure(figsize=(100,30))

plot_tree(t,filled=True,fontsize=60,feature_names=Features,class_names=["Control Campaign","Test Campaign"])
plt.savefig("Image3.png")
plt.show()





