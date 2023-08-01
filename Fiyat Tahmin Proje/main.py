import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("carstrain.csv")
selected_features = ['fueltype', 'doornumber', 'carbody', 'enginelocation', 'enginetype', 'cylindernumber', 'enginesize', 'horsepower', 'citympg', 'highwaympg', 'price']
data = data[selected_features]
data = pd.get_dummies(data, columns=['fueltype', 'doornumber', 'cylindernumber', 'carbody', 'enginelocation', 'enginetype'])

xData=data.drop(columns="price").values
yData=data["price"].values

xtrain,xtest,ytrain,ytest=train_test_split(xData,yData,test_size=0.2,random_state=42)
scaler=MinMaxScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)

from keras.models import Sequential
from keras.layers import Dense,Dropout

model=Sequential()
model.add(Dense(512,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.01))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.01))
model.add(Dense(32,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dropout(0.01))
model.add(Dense(8,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2,activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")
model.fit(x=xtrain,y=ytrain,validation_data=(xtest,ytest),batch_size=32,epochs=512)
tahmin=model.predict(xtest)
print(tahmin)
dizi=[i for i in range(40000)]
plt.plot(dizi,c="red")
plt.scatter(ytest,tahmin)
plt.savefig("ogrenmeGrafik.png")
plt.show()