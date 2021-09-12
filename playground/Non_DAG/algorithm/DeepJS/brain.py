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
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation='relu')
        self.dense_2 = tf.keras.layers.Dense(9, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(18, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(9, activation='relu')
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)

class BrainOneLayerLSTMActivationTanhFourLayersDenseActivationTanh(tf.keras.Model):
    name = 'BrainOneLayerLSTMActivationTanhFourLayersDenseActivationTanh'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.LSTM(3, input_shape=(state_size,1), activation=tf.tanh)
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

class BrainOneLayerLSTMActivationTanhFourLayersDenseActivationRelu(tf.keras.Model):
    name = 'BrainOneLayerLSTMActivationTanhFourLayersDenseActivationRelu'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.LSTM(3, input_shape=(state_size,1), , activation=tf.tanh)
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

class BrainOneLayerLSTMActivationReluFourLayersDenseActivationTanh(tf.keras.Model):
    name = 'BrainOneLayerLSTMActivationReluFourLayersDenseActivationTanh'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.LSTM(3, input_shape=(state_size,1), activation='relu')
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

class BrainOneLayerLSTMActivationReluFourLayersDenseActivationRelu(tf.keras.Model):
    name = 'BrainOneLayerLSTMActivationReluFourLayersDenseActivationRelu'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.LSTM(3, input_shape=(state_size,1), activation='relu')
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

class BrainOneLayerLSTMActivationSigmoidFourLayersDenseActivationTanh(tf.keras.Model):
    name = 'BrainOneLayerLSTMActivationSigmoidFourLayersDenseActivationTanh'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.LSTM(3, input_shape=(state_size,1), activation='sigmoid')
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

class BrainOneLayerLSTMActivationSigmoidFourLayersDenseActivationRelu(tf.keras.Model):
    name = 'BrainOneLayerLSTMActivationSigmoidFourLayersDenseActivationRelu'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.LSTM(3, input_shape=(state_size,1), activation='sigmoid')
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
 
class BrainOneLayerGRUActivationTanhFourLayersDenseActivationTanh(tf.keras.Model):
    name = 'BrainOneLayerGRUActivationTanhFourLayersDenseActivationTanh'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.GRU(3, input_shape=(state_size,1), activation=tf.tanh)
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

class BrainOneLayerGRUActivationTanhFourLayersDenseActivationRelu(tf.keras.Model):
    name = 'BrainOneLayerGRUActivationTanhFourLayersDenseActivationRelu'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.GRU(3, input_shape=(state_size,1), activation=tf.tanh)
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

class BrainOneLayerGRUActivationReluFourLayersDenseActivationTanh(tf.keras.Model):
    name = 'BrainOneLayerGRUActivationReluFourLayersDenseActivationTanh'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.GRU(3, input_shape=(state_size,1), activation='relu')
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

class BrainOneLayerGRUActivationReluFourLayersDenseActivationRelu(tf.keras.Model):
    name = 'BrainOneLayerGRUActivationReluFourLayersDenseActivationRelu'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.GRU(3, input_shape=(state_size,1), activation='relu')
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
 
class BrainOneLayerGRUActivationSigmoidFourLayersDenseActivationTanh(tf.keras.Model):
    name = 'BrainOneLayerGRUActivationSigmoidFourLayersDenseActivationTanh'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.GRU(3, input_shape=(state_size,1), activation='sigmoid')
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

class BrainOneLayerGRUActivationSigmoidFourLayersDenseActivationRelu(tf.keras.Model):
    name = 'BrainOneLayerGRUActivationSigmoidFourLayersDenseActivationRelu'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.GRU(3, input_shape=(state_size,1), activation='sigmoid')
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
