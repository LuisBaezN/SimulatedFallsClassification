from gc import callbacks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def norm(x, min, max):
    x_std = (x - min)/(max - min)
    s_scaled = x_std*(1 - 0) + 0
    return s_scaled

def build_model(in_shape):
    neurons=[300,150,70,1]
    model = keras.Sequential([
        layers.Dense(neurons[0], activation="relu", input_shape=[in_shape]),
        layers.Dense(neurons[1], activation="relu"),
        layers.Dense(neurons[2], activation="relu"),
        layers.Dense(neurons[3], activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
    return model

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

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

    model = build_model(data_ent_norm.shape[1])

    model.summary()

    # Verify model
    example_batch = data_ent_norm[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    epocas = 17    

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    training_h = model.fit(
        data_ent_norm, lab_ent,
        epochs=epocas, validation_split = 0.2, verbose=1,
        callbacks=[early_stop])

    #//////////////////////////// Results /////////////////////////////

    valid_predictions = model.predict(data_val_norm)

    print(valid_predictions)
    print(lab_val)

    #/////////////////////////// Training history /////////////////////

    plot_history(training_h)
