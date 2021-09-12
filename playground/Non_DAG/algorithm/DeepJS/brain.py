import tensorflow as tf
import numpy as np

class BrainFiveLayersDenseActivationTanh(tf.keras.Model):
    name = 'BrainFiveLayersDenseActivationTanh'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)

class BrainFiveLayersDenseActivationRelu(tf.keras.Model):
    name = 'BrainFiveLayersDenseActivationRelu'

    def __init__(self, state_size):
        super().__init__()
        #self.dense_1 = tf.keras.layers.GRU(3, input_shape=(state_size,1), activation='relu')
        #self.dense_2 = tf.keras.layers.LSTM(16, return_sequences=True)
        #self.dense_3 = tf.keras.layers.LSTM(16)
        #self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation='sigmoid')
        self.dense_2 = tf.keras.layers.Dense(9, input_shape=(None, state_size),activation='tanh')
        self.dense_3 = tf.keras.layers.Dense(18, activation='tanh')
        self.dense_4 = tf.keras.layers.Dense(9, activation='tanh')
        self.dense_1 = tf.keras.layers.GRU(3, input_shape=(state_size,1), activation='tanh')
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        #print("111")
        #print(np.shape(state))
        #state = np.reshape(state, (state.shape[0], state.shape[1], 1))
        #print("222")
        #print(np.shape(state))
        #state = self.dense_1(state)
        #print("333")
        #print(np.shape(state))
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        #print("666")
        state = np.reshape(state, (state.shape[0], state.shape[1], 1))
        state = self.dense_1(state)
        #print(np.shape(state))
        state = self.dense_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)

class BrainOneLayerLSTMFourLayersDenseActivationTanh(tf.keras.Model):
    name = 'BrainOneLayerLSTMFourLayersDenseActivationTanh'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.LSTM(3, input_shape=(state_size,1))
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = np.reshape(state, (state.shape[0], state.shape[1], 1))
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)

class BrainOneLayerLSTMFourLayersDenseActivationRelu(tf.keras.Model):
    name = 'BrainOneLayerLSTMFourLayersDenseActivationRelu'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.LSTM(3, input_shape=(state_size,1))
        self.dense_2 = tf.keras.layers.Dense(9, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(18, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(9, activation='relu')
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = np.reshape(state, (state.shape[0], state.shape[1], 1))
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)

class BrainOneLayerGRUFourLayersDenseActivationTanh(tf.keras.Model):
    name = 'BrainOneLayerGRUFourLayersDenseActivationTanh'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.GRU(3, input_shape=(state_size,1))
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = np.reshape(state, (state.shape[0], state.shape[1], 1))
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)

class BrainOneLayerGRUFourLayersDenseActivationRelu(tf.keras.Model):
    name = 'BrainOneLayerGRUFourLayersDenseActivationRelu'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.GRU(3, input_shape=(state_size,1))
        self.dense_2 = tf.keras.layers.Dense(9, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(18, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(9, activation='relu')
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = np.reshape(state, (state.shape[0], state.shape[1], 1))
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)
