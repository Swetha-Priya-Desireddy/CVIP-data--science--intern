# CVIP-data--science--intern
import pandas as pd
df=pd.read_csv("/content/Latest Covid-19 India Status.csv")
from google.colab import drive
drive.mount('/content/drive')
df
df.describe()
df.columns
df.median()
df.max()
df.min()
df.shape
df.info()
df.corr()
df.isnull()
a=df.head()
a
df.mean()
df.median()
df.mode()
df.head()
df.tail()
import matplotlib.pyplot as plt
plt.hist(df["Death Ratio"])
plt.show()
plt.bar(df["Death Ratio"],df["Total Cases"])
plt.show()
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 3.50]
line1, = plt.plot(df["Population"], label="Total population in India")
line2, = plt.plot(df["Total Cases"], label="Total cases in India")
leg = plt.legend(loc='upper center')
plt.show()
labels=["Andaman and Nicobar","Andhra Pradesh","Arunachal Pradesh","Assam","Bihar"]
plt.show()
fig, ax = plt.subplots()
ax.pie(a["Death Ratio"], labels=labels)
ax.set_title('Death Ratio of first 5 states')
plt.tight_layout()
import pandas as pd
print(df['Population'].quantile([0.25,0.5,0.75,1]))
boxplot=df.boxplot(column=['Population'],by='Death Ratio')
print(boxplot)
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)
print(train)
from sklearn.model_selection import train_test_split
y = df.pop('Death Ratio')
X = df
X_train,X_test,y_train,y_test = train_test_split(X.index,y,test_size=0.2)
X.iloc[X_train]
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
SVC(random_state=0)
predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()
