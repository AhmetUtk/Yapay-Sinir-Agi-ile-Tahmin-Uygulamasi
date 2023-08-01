import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("carstrain.csv")
selected_features = ['fueltype', 'doornumber', 'carbody', 'enginelocation', 'enginetype', 'cylindernumber', 'enginesize', 'horsepower', 'citympg', 'highwaympg', 'price']
data = data[selected_features]
data = pd.get_dummies(data, columns=['fueltype', 'doornumber', 'cylindernumber', 'carbody', 'enginelocation', 'enginetype'])

X = data.drop(columns=['price'])
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
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

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train_scaled, y_train, epochs=512, batch_size=32, validation_data=(X_test_scaled, y_test))

y_pred = model.predict(X_test_scaled).flatten()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Ortalama Kare Hata (Mean Squared Error):", mse)
print("R^2 Skoru:", r2)


plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("lossGrafik.png")
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)

preds_train = linear_model.predict(x_train_scaled)
train_r2 = r2_score(y_train, preds_train)
print("Linear Regression Eğitim R-kare skoru:", train_r2)

preds_test = linear_model.predict(x_test_scaled)
test_r2 = r2_score(y_test, preds_test)
print("Linear Regression Test R-kare skoru:", test_r2)
