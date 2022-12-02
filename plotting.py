import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_time_series(timesteps, values, format=".", start=0, end=None, label=None):
    """
    Plots a timesteps (a series of points in time) against values (a series of values across timesteps).

    Parameters
    ---------
    timesteps : array of timesteps
    values : array of values across time
    format : style of plot, default '.'
    start : where to start the plot (setting a value will index from start of timesteps & values)
    end : where to end the plot (setting a value will index from end of timesteps & values)
    label : label to show on plot of values
    """
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14)  # make label bigger
    plt.grid(True)


def plot_model_loss_training(model):
    model_history = pd.DataFrame(model.history.history)
    model_history["epoch"] = model.history.epoch

    fig, ax = plt.subplots(1, figsize=(8, 6))
    num_epochs = model_history.shape[0]

    ax.plot(np.arange(0, num_epochs), model_history["loss"], label="Training loss")
    ax.plot(
        np.arange(0, num_epochs), model_history["val_loss"], label="Validation loss"
    )
    ax.legend()
    plt.yscale(value="log")
    plt.tight_layout()
    plt.show()


def plot_prices_entire(training, validation, predictions):
    plt.figure(figsize=(16, 8))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close price USD", fontsize=18)
    plt.plot(training["price"])
    plt.plot(validation["price"])
    plt.plot(predictions["price"])
    plt.legend(["train", "val", "predictions"], loc="lower right")
    plt.show()


def plot_prices_per_return(training, validation):
    plt.figure(figsize=(16, 8))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close price USD", fontsize=18)
    plt.plot(training["price"])
    plt.plot(validation[["price", "predictions"]])
    plt.legend(["train", "val", "predictions"], loc="lower right")
    plt.show()


def plot_returns(training, validation, predictions):
    plt.figure(figsize=(16, 8))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Return, USD", fontsize=18)
    plt.plot(training["return"])
    plt.plot(validation["return"])
    plt.plot(predictions["predictions"])
    plt.legend(["train", "val", "predictions"], loc="lower right")
    plt.show()
