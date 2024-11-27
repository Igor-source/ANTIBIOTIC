# %%
import numpy as np
import tensorflow as tf
import numpy as np
import threading
# import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, losses, initializers


class CVAE(tf.keras.Model):
    def __init__(self, vocab_size, args):
        super(CVAE, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = args.batch_size
        self.latent_size = args.latent_size
        self.unit_size = args.unit_size
        self.num_prop = args.num_prop
        self.lr = args.lr
        self.stddev = args.stddev
        self.mean = args.mean

        # Embedding layers
        self.embedding_encode = layers.Embedding(
            input_dim=self.vocab_size, 
            output_dim=self.latent_size, 
            embeddings_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1)
        )
        self.embedding_decode = layers.Embedding(
            input_dim=self.vocab_size, 
            output_dim=self.latent_size, 
            embeddings_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1)
        )

        # Encoder RNN
        self.encoder = layers.RNN(
            [layers.LSTMCell(self.unit_size) for _ in range(3)], 
            return_sequences=True, 
            return_state=True
        )

        # Decoder RNN
        self.decoder = layers.RNN(
            [layers.LSTMCell(self.unit_size) for _ in range(3)], 
            return_sequences=True, 
            return_state=True
        )

        # Dense layers for latent space
        self.out_mean = layers.Dense(self.latent_size, kernel_initializer='glorot_uniform')
        self.out_log_sigma = layers.Dense(self.latent_size, kernel_initializer='glorot_uniform')

        # Softmax output layer
        self.softmax_layer = layers.Dense(self.vocab_size, activation='softmax')

        self.optimizer = optimizers.Adam(learning_rate=self.lr)

    def encode(self, X, C):
        X = self.embedding_encode(X)
        C = tf.expand_dims(C, 1)
        C = tf.tile(C, [1, tf.shape(X)[1], 1])
        inp = tf.concat([X, C], axis=-1)
        _, *state = self.encoder(inp)
        h = state[-1][1]  # Use hidden state
        mean = self.out_mean(h)
        log_sigma = self.out_log_sigma(h)
        eps = tf.random.normal([self.batch_size, self.latent_size], mean=0.0, stddev=1.0)
        latent_vector = mean + tf.exp(log_sigma / 2.0) * eps
        return latent_vector, mean, log_sigma

    def decode(self, Z, X, C, seq_length):
        Z = tf.expand_dims(Z, 1)
        Z = tf.tile(Z, [1, seq_length, 1])
        C = tf.expand_dims(C, 1)
        C = tf.tile(C, [1, seq_length, 1])
        X = self.embedding_decode(X)
        inputs = tf.concat([Z, X, C], axis=-1)
        outputs = self.decoder(inputs)
        logits = self.softmax_layer(outputs[0])
        return logits

    def call(self, inputs):
        X, Y, C, L = inputs
        latent_vector, mean, log_sigma = self.encode(X, C)
        decoded_logits = self.decode(latent_vector, X, C, tf.shape(X)[1])

        # Calculate losses
        mask = tf.sequence_mask(L, tf.shape(X)[1])
        weights = tf.cast(mask, dtype=tf.float32)
        reconstruction_loss = losses.sparse_categorical_crossentropy(Y, decoded_logits)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss * weights)

        latent_loss = -0.5 * tf.reduce_mean(1 + log_sigma - tf.square(mean) - tf.exp(log_sigma))

        loss = reconstruction_loss + latent_loss
        return loss, reconstruction_loss, latent_loss

    def train_step(self, data):
        X, Y, L, C = data
        with tf.GradientTape() as tape:
            loss, recon_loss, latent_loss = self((X, Y, C, L))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss, recon_loss, latent_loss
    
    def __init__(self, vocab_size, args):
        super(CVAE, self).__init__()
        # Инициализация слоев модели

    # def train_step(self, x, y, l, c):
    #     with tf.GradientTape() as tape:
    #         loss, recon_loss, latent_loss = self.call((x, y, c, l))
    #     grads = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    #     return loss.numpy()

    def test_step(self, x, y, l, c):
        loss, _, _ = self.call((x, y, c, l))
        return loss.numpy()
    def generate(self, latent_vector, C, start_token, seq_length):
        preds = []
        X = start_token
        for _ in range(seq_length):
            logits = self.decode(latent_vector, X, C, 1)
            X = tf.argmax(logits, axis=-1)
            preds.append(X)
        return tf.concat(preds, axis=1)
