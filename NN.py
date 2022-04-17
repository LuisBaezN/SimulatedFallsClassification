import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#//////////////////////Cargamos Dataset//////////////////////////////////
data = np.loadtxt('Fall_v2.csv', delimiter=',', skiprows = 1)

#//////////////////////Entrenamiento por porcentaje//////////////////////

por = .95
part = por*170 
part = int(part)

#///////////////////////////////////////////////////////////////////////
#Training data
atrib_ent = data[:part,0:292]
clas_ent = data[:part,293]

#Test data
atrib = data[part:,0:292]
clas = data[part:,293]

print('\n>Split data completed...\n')

#//////////////// Separamos Datos //////////////////////////////////////////////////

x_train, x_valid, y_train, y_valid = train_test_split(atrib_ent, clas_ent, test_size=0.20, shuffle= True, random_state= 3)

#/////////////// Normalizamos los datos ////////////////////////////////////////////

scaler = MinMaxScaler()
scaler.fit(x_train)
MinMaxScaler(copy=True, feature_range=(0,1))

x_train_norm = scaler.transform(x_train)

scaler.fit(x_valid)
MinMaxScaler(copy=True, feature_range=(0,1))

x_valid_norm = scaler.transform(x_valid)

#/////////////// Regression Model //////////////
def class_model():
    model = Sequential()
    model.add(Dense(300, activation ='relu', input_shape=(292,)))
    model.add(Dense(150, activation ='relu'))
    model.add(Dense(70, activation ='relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    return model

# build the model

model = class_model()

# fit the model
train_log = model.fit(x_train_norm, y_train, validation_data = (x_valid_norm, y_valid), batch_size = 3, epochs=7, verbose=1) 

#////////////////////////////////////////////////////////////////
#model.summary()

model.save('Classificador_bin_v1.h5') 
print(">Model Saved.")

#///////////////////////////////////Predict model//////////////////

scaler.fit(atrib)
MinMaxScaler(copy=True, feature_range=(0,1))

xfit = scaler.transform(atrib) #atrib
#Xfit = xfit[:,np.newaxis]

predictions = model.predict(xfit)

print(predictions)
print(clas)

plt.plot(train_log.history["loss"], label="loss")
plt.plot(train_log.history["val_loss"], label="val_loss")
plt.title("Traing")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.grid()
plt.legend()
plt.savefig("Entrenamiento_v1.png", dpi = 400)
plt.show()

predictions_red=np.round(predictions,2)

cm = confusion_matrix(clas, predictions_red)
print(cm)
