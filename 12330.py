import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
iris.feature_names

iris.target_names

df=pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

df['target']=iris.target
df.head()

df[df.target==1].head()

df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
df.head()

# Commented out IPython magic to ensure Python compatibility.
from matplotlib import pyplot as plt
# %matplotlib inline
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]
df0.head()

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='red',marker='*')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='black',marker='+')

plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='black',marker='*')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue',marker='.')

x=df[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
y=df.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35)

from sklearn.svm import SVC
model =SVC(kernel='linear')
model.fit(x_train,y_train)

model.score(x_test,y_test)

prediction=model.predict(x_test)

target_names = iris.target_names

for p in prediction:
    print(target_names[p], end = ', ')
