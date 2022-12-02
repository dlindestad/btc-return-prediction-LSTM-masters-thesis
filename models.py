import tensorflow as tf
from __main__ import WINDOW, n_features


def create_model(learning_rate, num_lstm_layers, num_lstm_nodes, dropout, n_features):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(WINDOW, n_features)))

    for i in range(num_lstm_layers):
        name = f"layer_lstm_{i + 1}"
        return_sequences = bool(i + 1 != num_lstm_layers)

        model.add(
            tf.keras.layers.LSTM(
                units=num_lstm_nodes,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=return_sequences,
                dropout=dropout,
                name=name,
            )
        )

    model.add(tf.keras.layers.Dense(units=1, activation="linear"))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mae")

    return model


model_dict = {}
model_dict["hp1"] = {
    "model": create_model(0.0006719412085277293, 1, 150, 0.0, n_features),
    "batch_size": 8,
}

model_dict["hp2"] = {
    "model": create_model(
        2.046523891190181e-05, 5, 158, 0.05624285327432276, n_features
    ),
    "batch_size": 8,
}

model_dict["hp3"] = {
    "model": create_model(
        9.699635032277529e-06, 5, 40, 0.06165566066185589, n_features
    ),
    "batch_size": 8,
}

model_dict["hp4"] = {
    "model": create_model(
        1.7140978536347443e-05, 5, 333, 0.14449588314562722, n_features
    ),
    "batch_size": 8,
}
model_dict["hp5"] = {
    "model": create_model(
        4.3703812920157626e-05, 5, 42, 0.14294847302317926, n_features
    ),
    "batch_size": 16,
}
model_dict["hp6"] = {
    "model": create_model(
        1.4195810963216943e-05, 4, 149, 0.2797002744958523, n_features
    ),
    "batch_size": 16,
}
model_dict["hp7"] = {
    "model": create_model(
        0.0007042393352265215, 5, 254, 0.26449868968864565, n_features
    ),
    "batch_size": 128,
}
model_dict["hp8"] = {
    "model": create_model(2.841005815734796e-05, 5, 182, 0.3, n_features),
    "batch_size": 8,
}
model_dict["hp9"] = {
    "model": create_model(4.391503277244555e-05, 5, 490, 0.3, n_features),
    "batch_size": 8,
}
model_dict["hp10"] = {
    "model": create_model(0.0005330971897217067, 5, 395, 0.3, n_features),
    "batch_size": 128,
}

model_dict["custom1"] = {
    "model": create_model(1e-6, 2, 400, 0.2, n_features),
    "batch_size": 128,
}
