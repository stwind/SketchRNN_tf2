import time
import numpy as np
import tensorflow as tf
from tensorflow import keras as K


class SketchRNN(object):
    def __init__(self, hps):
        self.hps = hps
        self.models = {}
        self.models["encoder"] = self._build_encoder()
        self.models["initial_state"] = self._build_initial_state()
        self.models["decoder"] = self._build_decoder()
        self.models["full"] = self._build_model()

    def _build_encoder(self):
        hps = self.hps
        encoder_input = K.layers.Input(
            shape=(hps["max_seq_len"], 5), name="encoder_input"
        )

        encoder_lstm_cell = K.layers.LSTM(
            units=hps["enc_rnn_size"], recurrent_dropout=hps["recurrent_dropout_prob"]
        )
        encoder_output = K.layers.Bidirectional(
            encoder_lstm_cell, merge_mode="concat", name="h"
        )(encoder_input)

        def reparameterize(z_params):
            mu, sigma = z_params
            sigma_exp = K.backend.exp(sigma / 2.0)
            return mu + sigma_exp * K.backend.random_normal(
                shape=K.backend.shape(sigma), mean=0.0, stddev=1.0
            )

        mu = K.layers.Dense(
            units=hps["z_size"],
            kernel_initializer=K.initializers.RandomNormal(stddev=0.001),
            name="mu",
        )(encoder_output)
        sigma = K.layers.Dense(
            units=hps["z_size"],
            kernel_initializer=K.initializers.RandomNormal(stddev=0.001),
            name="sigma",
        )(encoder_output)

        latent_z = K.layers.Lambda(reparameterize, name="z")([mu, sigma])

        return K.Model(
            inputs=encoder_input, outputs=[latent_z, mu, sigma], name="encoder"
        )

    def _build_initial_state(self):
        hps = self.hps
        z_input = K.layers.Input(shape=(hps["z_size"],), name="z_input")

        initial_state = K.layers.Dense(
            units=hps["dec_rnn_size"] * 2,
            activation="tanh",
            name="dec_initial_state",
            kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.001),
        )(z_input)
        states = tf.split(initial_state, 2, 1)

        return K.Model(inputs=z_input, outputs=states, name="initial_state")

    def _build_decoder(self):
        hps = self.hps
        decoder_input = K.layers.Input(shape=(None, 5), name="decoder_input")
        z_input = K.layers.Input(shape=(hps["z_size"],), name="z_input")
        initial_h_input = K.layers.Input(shape=(hps["dec_rnn_size"],), name="init_h")
        initial_c_input = K.layers.Input(shape=(hps["dec_rnn_size"],), name="init_c")

        decoder_lstm = K.layers.LSTM(
            units=hps["dec_rnn_size"],
            recurrent_dropout=hps["recurrent_dropout_prob"],
            name="decoder",
            return_sequences=True,
            return_state=True,
        )

        tile_z = tf.tile(tf.expand_dims(z_input, 1), [1, tf.shape(decoder_input)[1], 1])
        decoder_full_input = tf.concat([decoder_input, tile_z], -1)

        decoder_output, cell_h, cell_c = decoder_lstm(
            decoder_full_input, initial_state=[initial_h_input, initial_c_input]
        )

        output_layer = K.layers.Dense(units=hps["num_mixture"] * 6 + 3, name="output")
        output = output_layer(decoder_output)

        return K.Model(
            inputs=[decoder_input, z_input, initial_h_input, initial_c_input],
            outputs=[output, cell_h, cell_c],
            name="decoder",
        )

    def _build_model(self):
        hps = self.hps
        encoder_input = K.layers.Input(
            shape=(hps["max_seq_len"], 5), name="encoder_input"
        )
        decoder_input = K.layers.Input(shape=(None, 5), name="decoder_input")

        z_out, mu_out, sigma_out = self.models["encoder"](encoder_input)
        init_h, init_c = self.models["initial_state"](z_out)

        output, _, _ = self.models["decoder"]([decoder_input, z_out, init_h, init_c])

        return K.Model(
            inputs=[encoder_input, decoder_input],
            outputs=[output, mu_out, sigma_out],
            name="sketchrnn",
        )

    def load_weights(self, path):
        self.models["full"].load_weights(path)
        print("Loaded Weights From: {}".format(path))

    def sample(self, temperature=1.0, greedy=False, z=None):
        seq_len = self.hps["max_seq_len"]
        if z is None:
            z = np.random.randn(1, self.hps["z_size"]).astype("float32")

        prev_x = np.array([0, 0, 1, 0, 0], dtype=np.float32)
        cell_h, cell_c = self.models["initial_state"](z)

        strokes = np.zeros((seq_len, 5), dtype=np.float32)

        for i in range(seq_len):
            outouts, cell_h, cell_c = self.models["decoder"](
                [prev_x.reshape((1, 1, 5)), z, cell_h, cell_c]
            )

            o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen = get_mixture_coef(
                outouts
            )

            idx = get_pi_idx(o_pi[0, 0], temperature, greedy)
            idx_eos = get_pi_idx(o_pen[0, 0], temperature, greedy)

            next_x1, next_x2 = sample_gaussian_2d(
                o_mu1[0, 0, idx],
                o_mu2[0, 0, idx],
                o_sigma1[0, 0, idx],
                o_sigma2[0, 0, idx],
                o_corr[0, 0, idx],
                np.sqrt(temperature),
                greedy,
            )

            strokes[i] = [next_x1, next_x2, 0, 0, 0]
            strokes[i, idx_eos + 2] = 1

            prev_x = strokes[i]

        return strokes

    def train(
        self, initial_epoch, train_dataset, val_dataset, checkpoint, log_every=100
    ):
        hps = self.hps
        model = self.models["full"]
        optimizer = K.optimizers.Adam(
            learning_rate=hps["learning_rate"], clipvalue=hps["grad_clip"]
        )
        metrics = {
            n: K.metrics.Mean(n, dtype=tf.float32) for n in ["recon", "kl", "cost"]
        }

        kl_weight = K.backend.variable(hps["kl_weight_start"], name="kl_weight")

        step = initial_epoch * hps["num_batches"]

        @tf.function
        def train_step(inputs, target):
            with tf.GradientTape() as tape:
                outputs, mu, sigma = model(inputs)
                md_loss = K.backend.mean(calculate_md_loss(target, outputs))
                kl_loss = calculate_kl_loss(mu, sigma, hps["kl_tolerance"])
                total_loss = md_loss + kl_loss * kl_weight

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return md_loss, kl_loss, total_loss

        for epoch in range(initial_epoch + 1, hps["epochs"] + 1):
            start = time.time()

            K.backend.set_learning_phase(1)
            for batch, (inputs, target) in enumerate(train_dataset, 1):
                step += 1

                ## Update learning rate
                lr = (hps["learning_rate"] - hps["min_learning_rate"]) * hps[
                    "decay_rate"
                ] ** step + hps["min_learning_rate"]
                K.backend.set_value(optimizer.lr, K.backend.get_value(lr))

                ## update kl weight
                klw = (
                    hps["kl_weight"]
                    - (hps["kl_weight"] - hps["kl_weight_start"])
                    * hps["kl_decay_rate"] ** step
                )
                K.backend.set_value(kl_weight, K.backend.get_value(klw))

                md_loss, kl_loss, total_loss = train_step(inputs, target)

                if batch % log_every == 0:
                    msg = (
                        "[train] epoch: {}/{}, batch: {}, recon: {:.4f}, "
                        "kl: {:.4f}, cost: {:.4f}, lr: {:.6f}, klw: {:.4f}, time: {:.2f}"
                    )
                    print(
                        msg.format(
                            epoch,
                            hps["epochs"],
                            batch,
                            md_loss.numpy(),
                            kl_loss.numpy(),
                            total_loss.numpy(),
                            optimizer.learning_rate.numpy(),
                            kl_weight.numpy(),
                            time.time() - start,
                        )
                    )
                    start = time.time()

            K.backend.set_learning_phase(0)
            for inputs, target in val_dataset:
                outputs, mu, sigma = model(inputs)
                md_loss = K.backend.mean(calculate_md_loss(target, outputs))
                kl_loss = calculate_kl_loss(mu, sigma, hps["kl_tolerance"])
                total_loss = md_loss + kl_loss * kl_weight

                metrics["recon"](md_loss)
                metrics["kl"](kl_loss)
                metrics["cost"](total_loss)

            print(
                "[validate] epoch: {}/{}, recon: {:.4f}, kl: {:.4f}, cost: {:.4f}".format(
                    epoch,
                    hps["epochs"],
                    metrics["recon"].result(),
                    metrics["kl"].result(),
                    metrics["cost"].result(),
                )
            )

            model.save_weights(checkpoint.format(epoch, metrics["cost"].result()))

            for metric in metrics.values():
                metric.reset_states()


def get_mixture_coef(out_tensor):
    z_pen_logits = out_tensor[:, :, :3]
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(
        out_tensor[:, :, 3:], 6, 2
    )

    # Softmax all the pi's and pjen states:
    z_pi = K.activations.softmax(z_pi)
    z_pen = K.activations.softmax(z_pen_logits)

    # Exponent the sigmas and also make corr between -1 and 1.
    z_sigma1 = K.activations.exponential(z_sigma1)
    z_sigma2 = K.activations.exponential(z_sigma2)
    z_corr = K.activations.tanh(z_corr)

    return z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen


def keras_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    M = mu1.shape[2]  # Number of mixtures
    norm1 = K.backend.tile(K.backend.expand_dims(x1), [1, 1, M]) - mu1
    norm2 = K.backend.tile(K.backend.expand_dims(x2), [1, 1, M]) - mu2
    s1s2 = s1 * s2
    # eq 25
    z = (
        K.backend.square(norm1 / s1)
        + K.backend.square(norm2 / s2)
        - 2.0 * (rho * norm1 * norm2) / s1s2
    )
    neg_rho = 1.0 - K.backend.square(rho)
    result = K.backend.exp((-z) / (2 * neg_rho))
    denom = 2 * np.pi * s1s2 * K.backend.sqrt(neg_rho)
    result = result / denom
    return result


def calculate_kl_loss(mu, sigma, kl_tolerance):
    kl_cost = -0.5 * K.backend.mean(
        1 + sigma - K.backend.square(mu) - K.backend.exp(sigma)
    )
    return K.backend.maximum(kl_cost, kl_tolerance)


def calculate_md_loss(y_true, y_pred):
    o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen = get_mixture_coef(y_pred)

    x1_data, x2_data = y_true[:, :, 0], y_true[:, :, 1]
    pdf_values = keras_2d_normal(
        x1_data, x2_data, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr
    )

    gmm_values = pdf_values * o_pi
    gmm_values = K.backend.sum(gmm_values, 2, keepdims=True)

    epsilon = 1e-6
    gmm_loss = -K.backend.log(gmm_values + epsilon)  # avoid log(0)

    pen_data = y_true[:, :, 2:5]
    fs = 1.0 - pen_data[:, :, 2]
    fs = K.backend.expand_dims(fs)
    gmm_loss = gmm_loss * fs

    pen_loss = K.losses.categorical_crossentropy(pen_data, o_pen)
    pen_loss = K.backend.expand_dims(pen_loss)

    if not K.backend.learning_phase():
        pen_loss *= fs

    result = gmm_loss + pen_loss

    return result


def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf


def get_pi_idx(pdf, temp=1.0, greedy=False):
    if greedy:
        return np.argmax(pdf)
    a = np.arange(len(pdf))
    p = adjust_temp(pdf, temp)
    return np.random.choice(a, p=p)


def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
        return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0]

