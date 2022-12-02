from __main__ import np, pd, tf
import os
import json
import glob


def create_windows(arr, window_size, horizon=1):
    """
    Takes np array of 1 dimension, returns np array of windows 2d
    returns (windows, labels)
    """
    out = np.zeros(shape=(arr.shape[0] - (window_size + horizon - 1), window_size))
    label = np.zeros(shape=(arr.shape[0] - (window_size + horizon - 1), horizon))
    for i in range(arr.shape[0] - (window_size + horizon - 1)):
        out[i] = arr[i : i + window_size]
        label[i] = arr[i + window_size : i + window_size + horizon]
    return out, label


def create_train_test(features: dict, labels: dict, window=5, horizon=1, split=0.8):
    n_f = len(features.keys())
    n_l = len(labels.keys())
    f = np.array(list(features.values()))
    f = np.transpose(f, (1, 2, 0))
    # x_train
    xtr = f[: int(split * f.shape[0]), :, :]
    assert xtr.shape[0] >= (
        window + horizon - 1
    ), "Window and/or horizon size too large for given split."
    x_train = np.zeros((xtr.shape[0] - (window + horizon - 1), window, n_f))
    i = 0
    for feature in np.transpose(xtr, (2, 1, 0)):
        x_train[:, :, i], _ = create_windows(feature[0], window, horizon)
        i += 1
    # x_test
    xte = f[int(split * f.shape[0]) :, :, :]
    assert xte.shape[0] >= (
        window + horizon - 1
    ), "Window and/or horizon size too large for given split."
    x_test = np.zeros((xte.shape[0] - (window + horizon - 1), window, n_f))
    i = 0
    for feature in np.transpose(xte, (2, 1, 0)):
        x_test[:, :, i], _ = create_windows(feature[0], window, horizon)
        i += 1
    l = np.array(list(labels.values()))
    l = np.transpose(l, (1, 2, 0))
    # y_train
    ytr = l[: int(split * l.shape[0]), :, :]
    assert (
        ytr.shape[0] >= window + horizon
    ), "Window and or horizon too large for split."
    y_train = np.zeros((ytr.shape[0] - (window + horizon - 1), horizon, n_l))
    i = 0
    for label in np.transpose(ytr, (2, 1, 0)):
        _, y_train[:, :, i] = create_windows(label[0], window, horizon)
        i += 1
    # y_test
    yte = l[int(split * l.shape[0]) :, :, :]
    assert (
        yte.shape[0] >= window + horizon
    ), "Window and or horizon too large for split."
    y_test = np.zeros((yte.shape[0] - (window + horizon - 1), horizon, n_l))
    i = 0
    for label in np.transpose(yte, (2, 1, 0)):
        _, y_test[:, :, i] = create_windows(label[0], window, horizon)
        i += 1
    return x_train, y_train, x_test, y_test


def create_windows_full(features: dict, labels: dict, window=5, horizon=1):
    n_f = len(features.keys())
    n_l = len(labels.keys())
    f = np.array(list(features.values()))
    f = np.transpose(f, (1, 2, 0))
    x = np.zeros((f.shape[0] - (window + horizon - 1), window, n_f))
    i = 0
    for feature in np.transpose(f, (2, 1, 0)):
        x[:, :, i], _ = create_windows(feature[0], window, horizon)
        i += 1
    l = np.array(list(labels.values()))
    l = np.transpose(l, (1, 2, 0))
    y = np.zeros((l.shape[0] - (window + horizon - 1), horizon, n_l))
    i = 0
    for label in np.transpose(l, (2, 1, 0)):
        _, y[:, :, i] = create_windows(label[0], window, horizon)
        i += 1
    return x, y


def create_model_checkpoint(model_name, save_path="model_experiments"):
    """
    function to implement a ModelCheckpoint callback with a specific filename
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path, model_name),  # create filepath to save model
        verbose=0,  # only output a limited amount of text
        save_best_only=True,
    )  # save only the best model to file


def model_statistics(predictions_test, X_test, Y_test, model):
    mean_return = 0.0024243
    rmse = np.sqrt(np.mean((predictions_test - Y_test) ** 2))
    mse = np.mean((Y_test - predictions_test) ** 2)
    mae = np.mean(np.abs(Y_test - predictions_test))
    rmse_naive = np.sqrt(np.mean((np.zeros(predictions_test.shape) - Y_test) ** 2))
    mse_naive = np.mean((np.zeros(predictions_test.shape) - Y_test) ** 2)
    mae_naive = np.mean(np.abs(np.zeros(predictions_test.shape) - Y_test))
    rmse_er = np.sqrt(np.mean((np.full(Y_test.shape, mean_return) - Y_test) ** 2))
    mse_er = np.mean((np.full(Y_test.shape, mean_return) - Y_test) ** 2)
    mae_er = np.mean(np.abs(np.full(Y_test.shape, mean_return) - Y_test))
    error_statistics = {
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "rmse_naive": rmse_naive,
        "mse_naive": mse_naive,
        "mae_naive": mae_naive,
        "rmse_er": rmse_er,
        "mse_er": mse_er,
        "mae_er": mae_er,
    }
    return error_statistics


def print_model_statistics(error_statistics, X_test, Y_test, model):
    print("RMSE: ", error_statistics["rmse"])
    print("MSE: ", error_statistics["mse"])
    print("MAE: ", error_statistics["mae"])
    print("RMSE_naive: ", error_statistics["rmse_naive"])
    print("MSE_naive: ", error_statistics["mse_naive"])
    print("MAE_naive: ", error_statistics["mae_naive"])
    print("RMSE_expected_return: ", error_statistics["rmse_er"])
    print("MSE_expected_return: ", error_statistics["mse_er"])
    print("MAE_expected_return: ", error_statistics["mae_er"])
    # print(model.evaluate(X_test, Y_test))
    return


def generate_logreturns_csv():
    # BTC price (USD) bitstamp
    f_name = "Data/Market/bitcoinity_data_btc_price_usd_all_exchanges.csv"

    exchange_list = ["bitstamp", "coinbase", "gemini", "kraken"]
    df = pd.read_csv(
        f_name,
        index_col="Time",
        parse_dates=True,
        infer_datetime_format=True,
    )
    # YYYY-MM-DD HH:MM:SS+HH:MM to YYYY-MM-DD
    df.index = pd.to_datetime(df.index).date
    df = df.sort_index()
    start_date = pd.to_datetime("2012-01-01").date()
    end_date = pd.to_datetime("2022-09-01").date()
    daterange = pd.date_range(start=start_date, end=end_date).date

    # drop dates before start_date and after end_date

    df = df[~(df.index < start_date)]
    df = df[~(df.index > end_date)]

    # insert missing dates with NaN if any
    if len(df.index) != len(daterange):
        print(
            f'\nWarning: Dataset: "{f_name}" missing {len(daterange)-len(df.index)} days, inserting NaN for missing values.',
            "\nMissing dates (only first five are shown):",
        )
        collumn_dict_nan = {collumn_name: np.nan for collumn_name in df.columns}
        for i, date in enumerate(np.sort(list(set(daterange).difference(df.index)))):
            df = pd.concat([df, pd.DataFrame(collumn_dict_nan, index=[date])])
            if i < 5:
                print(date)
        print("\n")
        df = df.sort_index()

    assert len(df.index) == len(daterange)
    for column in df.columns:
        if column not in exchange_list:
            df = df.drop(columns=column)
    df["Mean"] = df.mean(axis=1)
    df_original_diff = df.copy()
    for exchange in exchange_list:
        df_original_diff[exchange + "_diff"] = (
            (df_original_diff[exchange] - df_original_diff["Mean"])
            / df_original_diff["Mean"]
        ) * 100
        df_original_diff = df_original_diff.drop(columns=exchange)
    df_original_diff = df_original_diff.drop(columns="Mean")

    df_clean = df.copy()
    df_clean = df_clean.drop(columns="Mean")
    s = 0

    outliers = set()
    for exchange in exchange_list:
        for idx, row in df_original_diff.iterrows():
            if row[exchange + "_diff"] >= 2:
                if sum(np.isnan(row.values)) < 3:
                    pass
                for exchange in exchange_list:
                    if df_original_diff.loc[idx, exchange + "_diff"] < 2:
                        outliers.add(idx)

    for outlier_idx in outliers:
        r_diff = list(df_original_diff.loc[outlier_idx].values)
        no_outliers = 0
        for value in r_diff:
            if value > 2:
                no_outliers += 1
        while no_outliers >= 1:
            max_outlier = max(r_diff, key=abs)
            max_outlier_idx = r_diff.index(max_outlier)
            df_clean.loc[outlier_idx, exchange_list[max_outlier_idx]] = np.nan
            r_diff[max_outlier_idx] = np.nan
            no_outliers -= 1
        df_original_diff.loc[idx].values

    import matplotlib.pyplot as plt

    df_clean["Mean"] = df_clean.mean(axis=1)
    df_clean = df_clean.sort_index()
    df_logreturns = pd.DataFrame(df_clean["Mean"])
    df_clean_a = np.array(df_clean["Mean"].values)
    log_r = np.diff(np.log(df_clean_a))
    log_r = np.insert(log_r, 0, 0)
    df_logreturns["Logreturn"] = log_r
    df_logreturns = df_logreturns.drop(columns="Mean")
    df_logreturns.to_csv("Data/Market/btc_logreturns_no_outliers.csv")
    return


def read_csv_bitcoinity(f_name: str, verbose: bool = False):
    start_date = pd.to_datetime("2012-01-01").date()
    end_date = pd.to_datetime("2022-09-01").date()
    daterange = pd.date_range(start=start_date, end=end_date).date
    df = pd.read_csv(
        f_name,
        index_col="Time",
        parse_dates=True,
        infer_datetime_format=True,
    )
    # YYYY-MM-DD HH:MM:SS+HH:MM to YYYY-MM-DD
    df.index = pd.to_datetime(df.index).date
    df = df.sort_index()

    # drop dates before start_date and after end_date
    df = df[~(df.index < start_date)]
    df = df[~(df.index > end_date)]

    # insert missing dates with NaN if any
    if len(df.index) != len(daterange):
        if verbose:
            print(
                f'\nWarning: Dataset: "{f_name}" missing {len(daterange)-len(df.index)} days, inserting NaN for missing values.',
                "\nMissing dates (only first five are shown):",
            )
        collumn_dict_nan = {collumn_name: np.nan for collumn_name in df.columns}
        for i, date in enumerate(np.sort(list(set(daterange).difference(df.index)))):
            df = pd.concat([df, pd.DataFrame(collumn_dict_nan, index=[date])])
            if verbose:
                if i < 5:
                    print(date)
        if verbose:
            print("\n")
        df = df.sort_index()

    assert len(df.index) == len(daterange)
    return df


def read_csv_blockchaincom(f_name: str, verbose: bool = False):
    start_date = pd.to_datetime("2012-01-01").date()
    end_date = pd.to_datetime("2022-09-01").date()
    daterange = pd.date_range(start=start_date, end=end_date).date
    df = pd.read_csv(
        f_name,
        index_col="x",
        parse_dates=True,
        infer_datetime_format=True,
    )

    df.index = pd.to_datetime(df.index, unit="ms").date
    df.index.name = "Date"

    # drop dates before start_date and after end_date
    df = df[~(df.index < start_date)]
    df = df[~(df.index > end_date)]

    # insert missing dates with NaN if any
    if len(df.index) != len(daterange):
        if verbose:
            print(
                f'\nWarning: Dataset: "{f_name}" missing {len(daterange)-len(df.index)} days, inserting NaN for missing values.',
                "\nMissing dates (only first five are shown):",
            )
        collumn_dict_nan = {collumn_name: np.nan for collumn_name in df.columns}
        for i, date in enumerate(np.sort(list(set(daterange).difference(df.index)))):
            df = pd.concat([df, pd.DataFrame(collumn_dict_nan, index=[date])])
            if verbose:
                if i < 5:
                    print(date)
        if verbose:
            print("\n")
        df = df.sort_index()

    assert len(df.index) == len(daterange)
    return df


def model_stats(model_name: str, pred_test, x_test, y_test, model, epoch_count):
    try:
        with open(
            os.path.normpath(os.path.join(os.getcwd(), "model_stats/modelstats.json")),
            "r",
        ) as f:
            model_performance = json.load(f)
    except FileNotFoundError:
        model_performance = {}

    model_performance[model_name] = model_statistics(pred_test, x_test, y_test, model)
    model_performance[model_name]["epoch_count"] = epoch_count
    with open(
        os.path.normpath(os.path.join(os.getcwd(), "model_stats/modelstats.json")), "w+"
    ) as f:
        json.dump(model_performance, f)
    return


def train_model(
    model, x_train, y_train, x_test, y_test, epochs, callbacks: list, batch_size=32
):
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        verbose=1,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )
    model.save("saved_models/modelbuilder_model.h5")
    return model


def clear_tmp_saved_models(folder_path: str):
    """
    Cleares temporary model checkpoints.
    """
    files = glob.glob(folder_path + "*")
    for f in files:
        assert ("model." in f) and (".h5" in f)
        try:
            os.remove(f)
        except PermissionError:
            print(f"Could not remove {f}, PermissionError. Skipping.")
    print("Cleaned up temporary model files.")
    return


def save_loss_history(num_epochs, train_loss, val_loss, name):
    loss_history = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "num_epochs": num_epochs,
    }
    df = pd.DataFrame.from_dict(loss_history)
    df.to_csv(f"model_stats/loss_history_{name}.csv")
    return


def main_plot(
    window,
    horizon,
    epochs,
    model,
    x_train,
    num_epochs,
    model_history,
    data,
    data_scalers,
    pred_test,
    tmp_fldr: str,
    plt_style: str,
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    df = pd.read_csv(
        "Data/Market/btc_logreturns_no_outliers.csv",
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    ax = fig.subplot_mosaic(
        [
            ["TopLeft", "TopRight", "Large", "Large"],
            ["BotLeft", "BotRight", "Large", "Large"],
        ],
    )

    ax["TopLeft"].plot(np.arange(0, num_epochs), model_history["loss"])
    ax["TopLeft"].plot(
        np.arange(0, num_epochs), model_history["val_loss"], label="Validation loss"
    )
    ax["TopLeft"].legend()
    ax["TopLeft"].set_yscale(value="log")
    ax["TopLeft"].set_title("Loss")

    x_full, _ = create_windows_full(
        data, {"btc_logreturn": data["btc_logreturn"]}, window=window, horizon=horizon
    )
    pred_full = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_full))
    ax["TopRight"].plot(df.index[window:], pred_full)
    ax["TopRight"].set_title("Full prediction")
    ax["TopRight"].set_ylim((-0.7, 0.7))

    ax["BotLeft"].plot(df["Logreturn"])
    ax["BotLeft"].set_title("Actual")
    ax["BotLeft"].set_ylim((-0.7, 0.7))

    ax["BotRight"].plot(df["Logreturn"][: x_train.shape[0] + window * 2])
    ax["BotRight"].plot(df.index[x_train.shape[0] + window * 2 :], pred_test)
    ax["BotRight"].set_title("Test prediction")
    ax["BotRight"].set_ylim((-0.7, 0.7))

    gs = ax["Large"].get_gridspec()
    ax["Large"].remove()
    ax["Large"] = fig.add_subplot(gs[:, 2:], projection="3d")
    interval_3d_plot = 5
    if plt_style == "complex":
        data_plot_3d = np.empty(shape=(round(epochs / 5) * len(pred_full), 3))

        for z in range(interval_3d_plot, epochs + interval_3d_plot, interval_3d_plot):
            model.load_weights(tmp_fldr + f"model.{z:04d}.h5")
            pred_line = data_scalers["btc_logreturn"].inverse_transform(
                model.predict(x_full)
            )
            for x, y in enumerate(pred_line):
                data_plot_3d[
                    (round(z / interval_3d_plot) - 1) * len(pred_full) + x, :
                ] = np.array([x, y[0], z])
        ax["Large"].plot_trisurf(
            data_plot_3d[:, 0],
            data_plot_3d[:, 1],
            data_plot_3d[:, 2],
            cmap="viridis",
            edgecolor="none",
        )
        ax["Large"].view_init(elev=33, azim=7, roll=95)
        ax["Large"].set_xlabel("X: Timestep (Date)")
        ax["Large"].set_ylabel("Y: Logreturn")
        ax["Large"].set_zlabel("Z: Epoch")
    elif plt_style == "simple":
        cmap = get_cmap("viridis_r")
        xdim_3d = np.arange(len(df.index[window:]))
        for z in range(interval_3d_plot, epochs + interval_3d_plot, interval_3d_plot):
            c = cmap(z / epochs)
            print(c)
            print((z / epochs + interval_3d_plot))
            try:
                model.load_weights(tmp_fldr + f"model.{z:04d}.h5")
            except FileNotFoundError:
                break
            pred_line = data_scalers["btc_logreturn"].inverse_transform(
                model.predict(x_full)
            )
            ax["Large"].plot(xdim_3d, pred_line, zs=z, zdir="z", c=c)
        ax["Large"].view_init(elev=33, azim=7, roll=95)
        ax["Large"].set_xlabel("X: Timestep (Date)")
        ax["Large"].set_ylabel("Y: Logreturn")
        ax["Large"].set_zlabel("Z: Epoch")
    fig.tight_layout()
    fig.savefig("Plots/main_plot_last.pdf")
    plt.show()
    return
