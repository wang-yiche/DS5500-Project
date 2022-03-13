import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Concatenate
from forecaster_tool.forecasters.mlp import MLPModel
# Code Source: https://www.kaggle.com/yamqwe/tutorial-time-series-transformer-time2vec/notebook#Submitting-To-Kaggle-%F0%9F%87%B0


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="gelu"), layers.Dense(feat_dim)])
        self.layernorm1 = layers.BatchNormalization()
        self.layernorm2 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb', shape=(input_shape[1],), initializer = 'uniform', trainable=True)
        self.bb = self.add_weight(name='bb', shape=(input_shape[1],), initializer = 'uniform', trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa', shape=(1, input_shape[1], self.k), initializer='uniform', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(1, input_shape[1], self.k), initializer='uniform', trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)
        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1] * (self.k + 1)))
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * (self.k + 1))


N_HEADS = 8
FF_DIM = 256
N_BLOCKS = 3
EMBED_DIM = 64
DROPUT_RATE = 0.0
SKIP_CONNECTION_STRENGTH = 0.9


def get_model(input_shape, forecast_length, time2vec_dim=3):
    inp = Input(input_shape)
    x = inp

    time_embedding = keras.layers.TimeDistributed(Time2Vec(time2vec_dim - 1))(x)
    x = Concatenate(axis=-1)([x, time_embedding])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    for k in range(N_BLOCKS):
        x_old = x
        transformer_block = TransformerBlock(EMBED_DIM, input_shape[-1] + (input_shape[-1] * time2vec_dim), N_HEADS, FF_DIM, DROPUT_RATE)
        x = transformer_block(x)
        x = ((1.0 - SKIP_CONNECTION_STRENGTH) * x) + (SKIP_CONNECTION_STRENGTH * x_old)

    x = layers.Flatten()(x)

    x = layers.Dense(128, activation="selu")(x)
    x = layers.Dropout(DROPUT_RATE)(x)
    x = Dense(forecast_length, activation='linear')(x)

    out = x
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mae')
    return model


class Transformer(MLPModel):

    def get_model_architecture(self):
        model = get_model(input_shape=(self.look_back_steps, self.X_train.shape[2]), forecast_length=self.forecast_steps)
        print(model.summary())
        return model
