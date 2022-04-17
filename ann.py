import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

def norm(x, min, max):
    x_std = (x - min)/(max - min)
    s_scaled = x_std*(1 - 0) + 0
    return s_scaled

def build_model():
    model = keras.Sequential([
        layers.Dense(300, activation="relu", input_shape=[270]),
        layers.Dense(150, activation="relu"),
        layers.Dense(70, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
    return model

if __name__ == "__main__":

    #////////////////////// Cargamos Dataset ////////////////////////////////
    data = np.loadtxt('Falls.csv', delimiter=',')

    print(data.shape)
    print(data.ndim)
    print(data.size)

    lim = 270

    #////////////////// Entrenamiento por porcentaje /////////////////////////

    por = 0.95
    part = por*data.shape[0]
    part = int(part)

    # Validation data
    data_val = data[part:,0:lim]
    lab_val = data[part:,data.shape[1]-1]

    # Base data
    data_ent = data[:part,0:lim]
    lab_ent = data[:part,data.shape[1] - 1]

    del data

    print('\n> Split data completed...\n')

    #////////////////////// Normalizar los datos ////////////////////////

    #print(data_ent.min())
    #print(data_ent.max())

    data_ent_norm = norm(data_ent, data_ent.min(), data_ent.max())
    data_val_norm = norm(data_val, data_val.min(), data_val.max())
    
    print(data_ent_norm.shape)
    
    '''
    plt.figure()
    plt.subplot(211)
    plt.plot(range(data_ent.shape[1]), data_ent[0])

    plt.subplot(212)
    plt.plot(range(data_ent_norm.shape[1]), data_ent_norm[0])
    plt.show()
    '''
    
    
    del data_ent
    
    #///////////////////////// Regression Model //////////////////////

    model = build_model()

    model.summary()

    # Verify model
    example_batch = data_ent_norm[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    epocas = 7    

    history = model.fit(
        data_ent_norm, lab_ent,
        epochs=epocas, validation_split = 0.2, verbose=1)

    #//////////////////////////// Results /////////////////////////////

    valid_predictions = model.predict(data_val_norm)

    print(valid_predictions)
    print(lab_val)