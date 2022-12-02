import numpy as np
import pandas as pd
import tensorflow as tf
from plotting import *
from functions import *
from import_data import data_scaled, data_scalers
import skopt

# current best:
# Search results:
# Learning rate: 2.5095534247216324e-05
# Number of lstm layers: 4
# Number of lstm units per layer: 473
# Best dropout value: 0.5877802924577188

# Fitness value: 0.02556636929512024


WINDOW = 50
HORIZON = 1
SPLIT = 0.8
EPOCHS = 20
OPTIMIZATION_ITERATIONS = 400

# define the hyperparameters that are searched
dim_learning_rate = skopt.space.Real(
    low=1e-6, high=1e-3, prior="log-uniform", name="learning_rate"
)
dim_num_lstm_layers = skopt.space.Integer(low=1, high=5, name="num_lstm_layers")
dim_num_lstm_nodes = skopt.space.Integer(low=5, high=512, name="num_lstm_nodes")
dim_dropout = skopt.space.Real(low=0.0, high=0.3, prior="uniform", name="dropout")
dim_batch_size = skopt.space.Categorical(
    categories=[8, 16, 32, 64, 128], name="batch_size"
)
dimensions = [
    dim_learning_rate,
    dim_num_lstm_layers,
    dim_num_lstm_nodes,
    dim_dropout,
    dim_batch_size,
]


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


@skopt.utils.use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_lstm_layers, num_lstm_nodes, dropout, batch_size):
    global fitness_glob
    x_train = fitness_glob["x_train"]
    y_train = fitness_glob["y_train"]
    x_test = fitness_glob["x_test"]
    y_test = fitness_glob["y_test"]
    n_features = fitness_glob["n_features"]

    # print the hyperparameters.
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_lstm_layers:", num_lstm_layers)
    print("num_lstm_nodes:", num_lstm_nodes)
    print("dropout:", dropout)
    print("batch size:", batch_size)
    print()

    # create the neural network with these hyper-parameters.
    model = create_model(
        learning_rate=learning_rate,
        num_lstm_layers=num_lstm_layers,
        num_lstm_nodes=num_lstm_nodes,
        dropout=dropout,
        n_features=n_features,
    )

    # Use Keras to train the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=15,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[],
    )

    loss = history.history["val_loss"][-1]
    global optimization_progress
    global lowest_loss
    print()
    print(f"Optimization progress: {optimization_progress}/{OPTIMIZATION_ITERATIONS}.")
    print(f"Loss: {loss}")
    print(f"Lowest loss: {lowest_loss}")
    print()
    optimization_progress += 1
    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    if loss < lowest_loss:
        lowest_loss = loss

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    tf.keras.backend.clear_session()
    return loss


def main():
    data = data_scaled
    x_train, y_train, x_test, y_test = create_train_test(
        data,
        {"btc_logreturn": data["btc_logreturn"]},
        window=WINDOW,
        horizon=HORIZON,
        split=SPLIT,
    )
    n_features = len(data.keys())

    global fitness_glob
    fitness_glob = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "n_features": n_features,
    }

    default_parameters = [1e-5, 1, 16, 0.2, 32]

    # define global variable to store loss
    global lowest_loss
    global optimization_progress
    optimization_progress = 0
    lowest_loss = 0
    lowest_loss = fitness(x=default_parameters)

    search_result = skopt.gp_minimize(
        func=fitness,
        dimensions=dimensions,
        acq_func="EI",  # Expected Improvement.
        n_calls=OPTIMIZATION_ITERATIONS,
        x0=default_parameters,
    )

    print("10 best results:")
    for result in sorted(zip(search_result.func_vals, search_result.x_iters))[0:10]:
        print(result)

    print("Search results:")
    print(f"Learning rate: {search_result.x[0]}")
    print(f"Number of lstm layers: {search_result.x[1]}")
    print(f"Number of lstm units per layer: {search_result.x[2]}")
    print(f"Best dropout value: {search_result.x[3]}")
    print(f"\nFitness value: {search_result.fun}")
    return


if __name__ == "__main__":
    main()
