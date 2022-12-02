import numpy as np
import pandas as pd
import tensorflow as tf
from plotting import *
from functions import *
from import_data import data_scaled, data_scalers, data_raw


WINDOW = 50
HORIZON = 1
SPLIT = 0.8
EPOCHS = 400
TRAIN_MODEL = True
ACTIVE_MODEL = "hp2"
TMP_FLDR = (
    "D:/Backup master data/models/tmp/"  # Temporary model files "saved_models/tmp/"
)


def main():
    ################### Data pre processing ######################
    data = data_scaled
    x_train, y_train, x_test, y_test = create_train_test(
        data,
        {"btc_logreturn": data["btc_logreturn"]},
        window=WINDOW,
        horizon=HORIZON,
        split=SPLIT,
    )

    print(y_test[:, 0, 0].shape)
    print(y_test[1:4, 0, 0])

    # define the model
    global n_features
    n_features = len(data.keys())
    from models import model_dict

    model = model_dict[ACTIVE_MODEL]["model"]

    model.build(input_shape=x_train.shape)
    print(model.summary())
    # stop early if model is not improving for patience=n epochs
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=25, restore_best_weights=True
    )
    ten_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=TMP_FLDR + "model.{epoch:04d}.h5",
        save_best_only=False,
        save_weights_only=True,
    )

    if TRAIN_MODEL:
        model = train_model(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            EPOCHS,
            [es_callback, ten_callback],
            batch_size=model_dict[ACTIVE_MODEL]["batch_size"],
        )
    else:
        model = tf.keras.models.load_model("saved_models/modelbuilder_model.h5")
        print("Model loaded from saved model (/saved_models/modelbuilder_model.h5).")

    epoch_count = len(model.history.epoch)
    model_history = pd.DataFrame(model.history.history)
    model_history["epoch"] = model.history.epoch
    num_epochs = model_history.shape[0]
    # plot_model_loss_training(model)

    # get the models predicted values
    # pred_train = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_train))
    pred_test = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_test))
    y_test_unscaled = data_scalers["btc_logreturn"].inverse_transform(y_test[:, :, 0])
    model_stats(model.name, pred_test, x_test, y_test_unscaled, model, epoch_count)
    print_model_statistics(
        model_statistics(pred_test, x_test, y_test, model), x_test, y_test, model
    )

    save_loss_history(
        num_epochs, model_history["loss"], model_history["val_loss"], "last_model"
    )
    main_plot(
        WINDOW,
        HORIZON,
        EPOCHS,
        model,
        x_train,
        num_epochs,
        model_history,
        data,
        data_scalers,
        pred_test,
        TMP_FLDR,
        "simple",
    )
    clear_tmp_saved_models(TMP_FLDR)
    return


if __name__ == "__main__":
    main()
