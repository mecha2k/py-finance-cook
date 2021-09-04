import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import os

from fastai import *
from fastai.tabular import *
from fastai.tabular.all import *
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset
from collections import OrderedDict
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from icecream import ic

plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 300
warnings.simplefilter(action="ignore", category=FutureWarning)


def performance_evaluation_report(model, show_plot=False, labels=None, show_pr_curve=False):
    """
    Function for creating a performance report of a classification model.
    Parameters
    ----------
    model : fastai Learner
        A trained model for Tabular data
    show_plot : bool
        Flag whether to show the plot
    labels : list
        List with the class names.
    show_pr_curve : bool
        Flag whether to also show the PR-curve. For this to take effect,
        show_plot must be True.
    Return
    ------
    stats : pd.Series
        A series with the most important evaluation metrics
    """
    preds_valid, y_test = model.get_preds(ds_type=DatasetType.Valid)
    y_pred = preds_valid.argmax(dim=-1)
    y_pred_prob = preds_valid.numpy()[:, 1]
    cm = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = (metrics.precision_score(y_test, y_pred),)
    recall = (metrics.recall_score(y_test, y_pred),)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)
    precision_vec, recall_vec, thresholds = metrics.precision_recall_curve(y_test, y_pred_prob)
    pr_auc = metrics.auc(recall_vec, precision_vec)

    if show_plot:
        if labels is None:
            labels = ["Negative", "Positive"]
        N_SUBPLOTS = 3 if show_pr_curve else 2
        PLOT_WIDTH = 15 if show_pr_curve else 12
        PLOT_HEIGHT = 5 if show_pr_curve else 6
        fig, ax = plt.subplots(1, N_SUBPLOTS, figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        fig.suptitle("Performance Evaluation", fontsize=16)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            linewidths=0.5,
            cmap="BuGn_r",
            square=True,
            cbar=False,
            ax=ax[0],
            annot_kws={"ha": "center", "va": "center"},
        )
        ax[0].set(xlabel="Predicted label", ylabel="Actual label", title="Confusion Matrix")
        ax[0].xaxis.set_ticklabels(labels)
        ax[0].yaxis.set_ticklabels(labels)
        ax[1].plot(fpr, tpr, "b-", label=f"ROC-AUC = {roc_auc:.2f}")
        ax[1].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        ax[1].plot(fp / (fp + tn), tp / (tp + fn), "ro", markersize=8, label="Decision Point")
        ax[1].plot([0, 1], [0, 1], "r--")
        ax[1].legend(loc="lower right")
        if show_pr_curve:
            ax[2].plot(recall_vec, precision_vec, label=f"PR-AUC = {pr_auc:.2f}")
            ax[2].plot(recall, precision, "ro", markersize=8, label="Decision Point")
            ax[2].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
            ax[2].legend()

    stats = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "specificity": (tn / (tn + fp)),
        "f1_score": metrics.f1_score(y_test, y_pred),
        "cohens_kappa": metrics.cohen_kappa_score(y_test, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    return stats


def create_input_data(series, n_lags=1, n_leads=1):
    """
    Function for transforming time series into input acceptable by a multilayer perceptron.
    Parameters
    ----------
    series : np.array
        The time series to be transformed
    n_lags : int
        The number of lagged observations to consider as features
    n_leads : int
        The number of future periods we want to forecast for
    Returns
    -------
    X : np.array
        Array of features
    y : np.array
        Array of target
    """
    X = []
    y = []
    for step in range(len(series) - n_lags - n_leads + 1):
        end_step = step + n_lags
        forward_end = end_step + n_leads
        X.append(series[step:end_step])
        y.append(series[end_step:forward_end])
    return np.array(X), np.array(y)


def custom_set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    custom_set_seed(42)
    # df = pd.read_csv("data/credit_card_default.csv", index_col=0, na_values="")
    # ic(df.head())
    #
    # DEP_VAR = "default_payment_next_month"
    # num_features = list(df.select_dtypes("number").columns)
    # num_features.remove(DEP_VAR)
    # cat_features = list(df.select_dtypes("object").columns)
    # preprocessing = [FillMissing, Categorify, Normalize]
    #
    # data = (
    #     TabularList.from_df(
    #         df, cat_names=cat_features, cont_names=num_features, procs=preprocessing
    #     )
    #     .split_by_rand_pct(valid_pct=0.2, seed=42)
    #     .label_from_df(cols=DEP_VAR)
    #     .databunch()
    # )
    # # We additionally inspect a few rows from the DataBunch:
    # data.show_batch(rows=5)
    #
    # # 5. Define the `Learner` object:
    # learn = tabular_learner(
    #     data,
    #     layers=[1000, 500],
    #     ps=[0.001, 0.01],
    #     emb_drop=0.04,
    #     metrics=[Recall(), FBeta(beta=1), FBeta(beta=5)],
    # )
    # # 6. Inspect the model's architecture:
    # ic(learn.model)
    #
    # # `Embedding(11, 6)` means that a categorical embedding was created with 11 input values and 6 output latent features.
    # learn.lr_find()
    # learn.recorder.plot(suggestion=True)
    # plt.tight_layout()
    # plt.savefig("images/ch10_im2.png")
    #
    # learn.fit(epochs=25, lr=1e-6, wd=0.2)
    # learn.recorder.plot_losses()
    # plt.tight_layout()
    # plt.savefig("images/ch10_im4.png")
    #
    # preds_valid, _ = learn.get_preds(ds_type=DatasetType.Valid)
    # pred_valid = preds_valid.argmax(dim=-1)
    #
    # # 11. Inspect the performance (confusion matrix) on the validation set:
    # interp = ClassificationInterpretation.from_learner(learn)
    # interp.plot_confusion_matrix()
    # plt.tight_layout()
    # plt.savefig("images/ch10_im5.png")
    # interp.plot_tab_top_losses(5)
    # performance_evaluation_report(learn)
    #
    # X = df.copy()
    # y = X.pop(DEP_VAR)
    # train_ind, test_ind = next(StratifiedKFold(n_splits=5).split(X, y))
    # data = (
    #     TabularList.from_df(
    #         df, cat_names=cat_features, cont_names=num_features, procs=preprocessing
    #     )
    #     .split_by_idxs(train_idx=list(train_ind), valid_idx=list(test_ind))
    #     .label_from_df(cols=DEP_VAR)
    #     .databunch()
    # )

    ## Multilayer perceptrons for time series forecasting
    print(torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    TICKER = "ANF"
    START_DATE = "2010-01-02"
    END_DATE = "2019-12-31"
    N_LAGS = 3

    # neural network
    VALID_SIZE = 12
    BATCH_SIZE = 16
    N_EPOCHS = 1000

    df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    df = df.resample("M").last()
    prices = df["Adj Close"].values

    fig, ax = plt.subplots()
    ax.plot(df.index, prices)
    ax.set(title=f"{TICKER}'s Stock price", xlabel="Time", ylabel="Price ($)")
    plt.tight_layout()
    plt.savefig("images/ch10_im6.png")

    def create_input_data(series, n_lags=1):
        """
        Function for transforming time series into input acceptable by a multilayer perceptron.
        Parameters
        ----------
        series : np.array
            The time series to be transformed
        n_lags : int
            The number of lagged observations to consider as features
        Returns
        -------
        X : np.array
            Array of features
        y : np.array
            Array of target
        """
        X, y = [], []
        for step in range(len(series) - n_lags):
            end_step = step + n_lags
            X.append(series[step:end_step])
            y.append(series[end_step])
        return np.array(X), np.array(y)

    # 5. Transform the considered time series into input for the MLP:
    X, y = create_input_data(prices, N_LAGS)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().unsqueeze(dim=1)

    # 6. Create training and validation sets:
    valid_ind = len(X) - VALID_SIZE
    dataset = TensorDataset(X_tensor, y_tensor)
    train_dataset = Subset(dataset, list(range(valid_ind)))
    valid_dataset = Subset(dataset, list(range(valid_ind, len(X))))

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)

    # Inspect the observations from the first batch:
    ic(next(iter(train_loader))[0])
    ic(next(iter(train_loader))[1])

    # Check the size of the datasets:
    print(
        f"Size of datasets - training: {len(train_loader.dataset)} | validation: {len(valid_loader.dataset)}"
    )

    # 7. Use naive forecast as a benchmark and evaluate the performance:
    naive_pred = prices[len(prices) - VALID_SIZE - 1 : -1]
    y_valid = prices[len(prices) - VALID_SIZE :]

    naive_mse = mean_squared_error(y_valid, naive_pred)
    naive_rmse = np.sqrt(naive_mse)
    print(f"Naive forecast - MSE: {naive_mse:.2f}, RMSE: {naive_rmse:.2f}")

    # BONUS: Testing Linear Regression
    X_train = X[
        :valid_ind,
    ]
    y_train = y[:valid_ind]
    X_valid = X[
        valid_ind:,
    ]
    y_valid = y[valid_ind:]

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_valid)
    lr_mse = mean_squared_error(y_valid, y_pred)
    lr_rmse = np.sqrt(lr_mse)
    print(f"Linear Regression's forecast - MSE: {lr_mse:.2f}, RMSE: {lr_rmse:.2f}")
    print(f"Linear Regression's coefficients: {lin_reg.coef_}")

    fig, ax = plt.subplots()
    ax.plot(y_valid, color="blue", label="Actual")
    ax.plot(y_pred, color="red", label="Prediction")
    ax.set(title="Linear Regression's Forecasts", xlabel="Time", ylabel="Price ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/ch10_im6_1.png")

    class MLP(nn.Module):
        def __init__(self, input_size):
            super(MLP, self).__init__()
            self.linear1 = nn.Linear(input_size, 8)
            self.linear2 = nn.Linear(8, 4)
            self.linear3 = nn.Linear(4, 1)
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, x):
            x = self.linear1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.linear3(x)
            return x

    torch.manual_seed(42)
    model = MLP(N_LAGS).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ic(model)

    PRINT_EVERY = 50
    train_losses, valid_losses = [], []
    for epoch in range(N_EPOCHS):
        running_loss_train = 0
        running_loss_valid = 0
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_hat = model(x_batch)
            loss = loss_fn(y_batch, y_hat)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item() * x_batch.size(0)
        epoch_loss_train = running_loss_train / len(train_loader.dataset)
        train_losses.append(epoch_loss_train)
        with torch.no_grad():
            model.eval()
            for x_val, y_val in valid_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_hat = model(x_val)
                loss = loss_fn(y_val, y_hat)
                running_loss_valid += loss.item() * x_val.size(0)

            epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)

            if epoch > 0 and epoch_loss_valid < min(valid_losses):
                best_epoch = epoch
                torch.save(model.state_dict(), "data/mlp_checkpoint.pth")

            valid_losses.append(epoch_loss_valid)

        if epoch % PRINT_EVERY == 0:
            print(
                f"<{epoch}> - Train. loss: {epoch_loss_train:.2f} \t Valid. loss: {epoch_loss_valid:.2f}"
            )
    print(f"Lowest loss recorded in epoch: {best_epoch}")

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots()
    ax.plot(train_losses, color="blue", label="Training loss")
    ax.plot(valid_losses, color="red", label="Validation loss")
    ax.set(title="Loss over epochs", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/ch10_im7.png")

    # 12. Load the best model (with the lowest validation loss):
    state_dict = torch.load("data/mlp_checkpoint.pth")
    model.load_state_dict(state_dict)
    # 13. Obtain the predictions:
    y_pred, y_valid = [], []
    with torch.no_grad():
        model.eval()
        for x_val, y_val in valid_loader:
            x_val = x_val.to(device)
            y_pred.append(model(x_val))
            y_valid.append(y_val)
    y_pred = torch.cat(y_pred).cpu().numpy().flatten()
    y_valid = torch.cat(y_valid).cpu().numpy().flatten()

    # 14. Evaluate the predictions:
    mlp_mse = mean_squared_error(y_valid, y_pred)
    mlp_rmse = np.sqrt(mlp_mse)
    print(f"MLP's forecast - MSE: {mlp_mse:.2f}, RMSE: {mlp_rmse:.2f}")

    fig, ax = plt.subplots()
    ax.plot(y_valid, color="blue", label="True")
    ax.plot(y_pred, color="red", label="Prediction")
    ax.set(title="Multilayer Perceptron's Forecasts", xlabel="Time", ylabel="Price ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/ch10_im8.png")

    #### A sequential approach to defining the network's architecture
    # Below we define the same network as we have already used before in this recipe:
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(4, 1),
    )
    ic(model)

    #### Estimating neural networks using `scikit-learn`
    #
    # # 1. Import the libraries:
    #
    # # In[25]:
    #
    #
    # from sklearn.neural_network import MLPRegressor
    #
    #
    # # 2. Define the MLP using scikit-learn:
    #
    # # In[26]:
    #
    #
    # mlp = MLPRegressor(hidden_layer_sizes=(8, 4,),
    #                    learning_rate='constant',
    #                    batch_size=5,
    #                    max_iter=1000,
    #                    random_state=42)
    #
    #
    # # 3. Split the data into training and test set:
    #
    # # In[27]:
    #
    #
    # valid_ind = len(X) - VALID_SIZE
    #
    # X_train = X[:valid_ind, ]
    # y_train = y[:valid_ind]
    #
    # X_valid = X[valid_ind:, ]
    # y_valid = y[valid_ind:]
    #
    #
    # # 4. Train the MLP:
    #
    # # In[28]:
    #
    #
    # mlp.fit(X_train, y_train)
    #
    #
    # # 5. Plot the loss function over epochs:
    #
    # # In[29]:
    #
    #
    # plt.plot(mlp.loss_curve_)
    #
    #
    # # 6. Obtain the predictions:
    #
    # # In[30]:
    #
    #
    # y_pred = mlp.predict(X_valid)
    #
    #
    # # 7. Evaluate the predictions and plot them versus the observed values:
    #
    # # In[31]:
    #
    #
    # sk_mlp_mse = mean_squared_error(y_valid, y_pred)
    # sk_mlp_rmse = np.sqrt(sk_mlp_mse)
    # print(f"Scikit-Learn MLP's forecast - MSE: {sk_mlp_mse:.2f}, RMSE: {sk_mlp_rmse:.2f}")
    #
    # fig, ax = plt.subplots()
    #
    # ax.plot(y_valid, color='blue', label='Actual')
    # ax.plot(y_pred, color='red', label='Prediction')
    #
    # ax.set(title="sklearn MLP's Forecasts",
    #        xlabel='Time',
    #        ylabel='Price ($)')
    # ax.legend();
    #
    #
    # # #### Multi-period forecast
    #
    # # 1. Define a modified function for creating a dataset for the MLP:
    #
    # # In[32]:
    #
    #
    # def create_input_data(series, n_lags=1, n_leads=1):
    #     '''
    #     Function for transforming time series into input acceptable by a multilayer perceptron.
    #
    #     Parameters
    #     ----------
    #     series : np.array
    #         The time series to be transformed
    #     n_lags : int
    #         The number of lagged observations to consider as features
    #     n_leads : int
    #         The number of future periods we want to forecast for
    #
    #     Returns
    #     -------
    #     X : np.array
    #         Array of features
    #     y : np.array
    #         Array of target
    #     '''
    #     X, y = [], []
    #
    #     for step in range(len(series) - n_lags - n_leads + 1):
    #         end_step = step + n_lags
    #         forward_end = end_step + n_leads
    #         X.append(series[step:end_step])
    #         y.append(series[end_step:forward_end])
    #     return np.array(X), np.array(y)
    #
    #
    # # 2. Create features and target from the time series of prices:
    #
    # # In[33]:
    #
    #
    # # parameters for the dataset
    # N_LAGS = 3
    # N_FUTURE = 2
    #
    # X, y = create_input_data(prices, N_LAGS, N_FUTURE)
    #
    # X_tensor = torch.from_numpy(X).float()
    # y_tensor = torch.from_numpy(y).float()
    #
    #
    # # 3. Create training and validation sets:
    #
    # # In[34]:
    #
    #
    # dataset = TensorDataset(X_tensor, y_tensor)
    #
    # valid_ind = len(X) - VALID_SIZE + (N_FUTURE - 1)
    #
    # train_dataset = Subset(dataset, list(range(valid_ind)))
    # valid_dataset = Subset(dataset, list(range(valid_ind, len(X))))
    #
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=BATCH_SIZE)
    # valid_loader = DataLoader(dataset=valid_dataset,
    #                           batch_size=BATCH_SIZE)
    #
    #
    # # 4. Define the MLP for multi-period forecasting:
    #
    # # In[35]:
    #
    #
    # class MLP(nn.Module):
    #
    #     def __init__(self, input_size, output_size):
    #         super(MLP, self).__init__()
    #         self.linear1 = nn.Linear(input_size, 16)
    #         self.linear2 = nn.Linear(16, 8)
    #         self.linear3 = nn.Linear(8, output_size)
    #         self.dropout = nn.Dropout(p=0.2)
    #
    #     def forward(self, x):
    #         x = self.linear1(x)
    #         x = F.relu(x)
    #         x = self.dropout(x)
    #         x = self.linear2(x)
    #         x = F.relu(x)
    #         x = self.dropout(x)
    #         x = self.linear3(x)
    #         return x
    #
    #
    # # 5. Instantiate the model, the loss function and the optimizer:
    #
    # # In[36]:
    #
    #
    # # set seed for reproducibility
    # torch.manual_seed(42)
    #
    # model = MLP(N_LAGS, N_FUTURE).to(device)
    # loss_fn = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    #
    # # 6. Train the network:
    #
    # # In[37]:
    #
    #
    # PRINT_EVERY = 50
    # train_losses, valid_losses = [], []
    #
    # for epoch in range(N_EPOCHS):
    #     running_loss_train = 0
    #     running_loss_valid = 0
    #
    #     model.train()
    #
    #     for x_batch, y_batch in train_loader:
    #
    #         optimizer.zero_grad()
    #
    #         x_batch = x_batch.to(device)
    #         y_batch = y_batch.to(device)
    #         y_hat = model(x_batch)
    #         loss = loss_fn(y_batch, y_hat)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss_train += loss.item() * x_batch.size(0)
    #
    #     epoch_loss_train = running_loss_train / len(train_loader.dataset)
    #     train_losses.append(epoch_loss_train)
    #
    #     with torch.no_grad():
    #
    #         model.eval()
    #
    #         for x_val, y_val in valid_loader:
    #             x_val = x_val.to(device)
    #             y_val = y_val.to(device)
    #             y_hat = model(x_val)
    #             loss = loss_fn(y_val, y_hat)
    #             running_loss_valid += loss.item() * x_val.size(0)
    #
    #         epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)
    #
    #         if epoch > 0 and epoch_loss_valid < min(valid_losses):
    #             best_epoch = epoch
    #             torch.save(model.state_dict(), './mlp_checkpoint_2.pth')
    #
    #         valid_losses.append(epoch_loss_valid)
    #
    #     if epoch % PRINT_EVERY == 0:
    #         print(f"<{epoch}> - Train. loss: {epoch_loss_train:.2f} \t Valid. loss: {epoch_loss_valid:.2f}")
    #
    # print(f'Lowest loss recorded in epoch: {best_epoch}')
    #
    #
    # # 7. Plot the training and validation losses:
    #
    # # In[39]:
    #
    #
    # train_losses = np.array(train_losses)
    # valid_losses = np.array(valid_losses)
    #
    # fig, ax = plt.subplots()
    #
    # ax.plot(train_losses, color='blue', label='Training loss')
    # ax.plot(valid_losses, color='red', label='Validation loss')
    #
    # ax.set(title="Loss over epochs",
    #        xlabel='Epoch',
    #        ylabel='Loss')
    # ax.legend();
    #
    #
    # # 8. Load the best model (with the lowest validation loss):
    #
    # # In[40]:
    #
    #
    # state_dict = torch.load('mlp_checkpoint_2.pth')
    # model.load_state_dict(state_dict)
    #
    #
    # # 9. Obtain predictions:
    #
    # # In[41]:
    #
    #
    # y_pred = []
    #
    # with torch.no_grad():
    #
    #     model.eval()
    #
    #     for x_val, y_val in valid_loader:
    #         x_val = x_val.to(device)
    #         yhat = model(x_val)
    #         y_pred.append(yhat)
    #
    # y_pred = torch.cat(y_pred).numpy()
    #
    #
    # # 10. Plot the predictions:
    #
    # # In[42]:
    #
    #
    # fig, ax = plt.subplots()
    #
    # ax.plot(y_valid, color='blue', label='Actual')
    #
    # for i in range(len(y_pred)):
    #     if i == 0:
    #         ax.plot(np.array([i, i + 1]), y_pred[i], color='red', label='Prediction')
    #     else:
    #         ax.plot(np.array([i, i + 1]), y_pred[i], color='red')
    #
    # ax.set(title="MLP's Multi-period Forecasts",
    #        xlabel='Time',
    #        ylabel='Price ($)')
    # ax.legend()
    #
    # # plt.tight_layout()
    # # plt.savefig('images/ch10_im9.png')
    # plt.show()
    #
    #

    ## Convolutional neural networks for time series forecasting
    print(torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ic(device)

    TICKER = "INTC"
    START_DATE = datetime(2010, 1, 2)
    END_DATE = datetime(2019, 12, 31)
    VALID_START = datetime(2019, 7, 1)
    N_LAGS = 12

    BATCH_SIZE = 256
    N_EPOCHS = 2000

    src_data = "data/yf_intl.pkl"
    try:
        data = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        data.to_pickle(src_data)
    df = data
    df = df.resample("W-MON").last()
    valid_size = df.loc[VALID_START:END_DATE].shape[0]
    prices = df["Adj Close"].values.reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.plot(df.index, prices)
    ax.set(title=f"{TICKER}'s Stock price", xlabel="Time", ylabel="Price ($)")

    # 4. Transform the time series into input for the CNN:
    X, y = create_input_data(prices, N_LAGS)

    # 5. Obtain the naïve forecast:
    naive_pred = prices[len(prices) - valid_size - 1 : -1]
    y_valid = prices[len(prices) - valid_size :]

    naive_mse = mean_squared_error(y_valid, naive_pred)
    naive_rmse = np.sqrt(naive_mse)
    print(f"Naive forecast - MSE: {naive_mse:.2f}, RMSE: {naive_rmse:.2f}")

    # 6. Prepare the `DataLoader` objects:
    custom_set_seed(42)
    valid_ind = len(X) - valid_size
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().unsqueeze(dim=1)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_dataset = Subset(dataset, list(range(valid_ind)))
    valid_dataset = Subset(dataset, list(range(valid_ind, len(X))))

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)

    # Check the size of the datasets:
    print(
        f"Size of datasets - training: {len(train_loader.dataset)} | validation: {len(valid_loader.dataset)}"
    )

    class Flatten(nn.Module):
        @staticmethod
        def forward(x):
            return x.view(x.size()[0], -1)

    model = nn.Sequential(
        OrderedDict(
            [
                ("conv_1", nn.Conv1d(1, 32, 3, padding=1)),
                ("max_pool_1", nn.MaxPool1d(2)),
                ("relu_1", nn.ReLU()),
                ("flatten", Flatten()),
                ("fc_1", nn.Linear(192, 50)),
                ("relu_2", nn.ReLU()),
                ("dropout_1", nn.Dropout(0.4)),
                ("fc_2", nn.Linear(50, 1)),
            ]
        )
    )
    print(model)

    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    PRINT_EVERY = 50
    train_losses, valid_losses = [], []
    for epoch in range(N_EPOCHS):
        running_loss_train = 0
        running_loss_valid = 0
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            x_batch = x_batch.view(x_batch.shape[0], 1, N_LAGS)
            y_batch = y_batch.to(device)
            y_batch = y_batch.view(y_batch.shape[0], 1, 1)
            y_hat = model(x_batch).view(y_batch.shape[0], 1, 1)
            loss = torch.sqrt(loss_fn(y_batch, y_hat))
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item() * x_batch.size(0)
        epoch_loss_train = running_loss_train / len(train_loader.dataset)
        train_losses.append(epoch_loss_train)

        with torch.no_grad():
            model.eval()
            for x_val, y_val in valid_loader:
                x_val = x_val.to(device)
                x_val = x_val.view(x_val.shape[0], 1, N_LAGS)
                y_val = y_val.to(device)
                y_val = y_val.view(y_val.shape[0], 1, 1)
                y_hat = model(x_val).view(y_val.shape[0], 1, 1)
                loss = torch.sqrt(loss_fn(y_val, y_hat))
                running_loss_valid += loss.item() * x_val.size(0)
            epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)
            if epoch > 0 and epoch_loss_valid < min(valid_losses):
                best_epoch = epoch
                torch.save(model.state_dict(), "data/cnn_checkpoint.pth")
            valid_losses.append(epoch_loss_valid)
        if epoch % PRINT_EVERY == 0:
            print(
                f"<{epoch}> - Train. loss: {epoch_loss_train:.6f} \t Valid. loss: {epoch_loss_valid:.6f}"
            )
    print(f"Lowest loss recorded in epoch: {best_epoch}")

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots()
    ax.plot(train_losses, color="blue", label="Training loss")
    ax.plot(valid_losses, color="red", label="Validation loss")

    ax.set(title="Loss over epochs", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/ch10_im11.png")

    # 11. Load the best model (with the lowest validation loss):
    state_dict = torch.load("data/cnn_checkpoint.pth")
    model.load_state_dict(state_dict)
    # 12. Obtain the predictions:
    y_pred, y_valid = [], []
    with torch.no_grad():
        model.eval()
        for x_val, y_val in valid_loader:
            x_val = x_val.to(device)
            x_val = x_val.view(x_val.shape[0], 1, N_LAGS)
            y_pred.append(model(x_val))
            y_valid.append(y_val)
    y_pred = torch.cat(y_pred).cpu().numpy().flatten()
    y_valid = torch.cat(y_valid).cpu().numpy().flatten()

    # 13. Evaluate the predictions:
    cnn_mse = mean_squared_error(y_valid, y_pred)
    cnn_rmse = np.sqrt(cnn_mse)
    print(f"CNN's forecast - MSE: {cnn_mse:.2f}, RMSE: {cnn_rmse:.2f}")

    fig, ax = plt.subplots()
    ax.plot(y_valid, color="blue", label="Actual")
    ax.plot(y_pred, color="red", label="Prediction")
    ax.plot(naive_pred, color="green", label="Naïve")
    ax.set(title="CNN's Forecasts", xlabel="Time", ylabel="Price ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/ch10_im12.png")

    ## Recurrent neural networks for time series forecasting
    BATCH_SIZE = 32
    N_EPOCHS = 100

    # 4. Scale the time series of prices:
    valid_ind = len(prices) - valid_size
    minmax = MinMaxScaler(feature_range=(0, 1))

    prices_train = prices[:valid_ind]
    prices_valid = prices[valid_ind:]

    minmax.fit(prices_train)

    prices_train = minmax.transform(prices_train)
    prices_valid = minmax.transform(prices_valid)

    prices_scaled = np.concatenate((prices_train, prices_valid)).flatten()
    plt.clf()
    plt.plot(prices_scaled)
    plt.tight_layout()
    plt.savefig("images/ch10_im13.png")

    # 5. Transform the time series into input for the RNN:
    X, y = create_input_data(prices_scaled, N_LAGS)

    # 6. Obtain the naïve forecast:
    naive_pred = prices[len(prices) - valid_size - 1 : -1]
    y_valid = prices[len(prices) - valid_size :]

    naive_mse = mean_squared_error(y_valid, naive_pred)
    naive_rmse = np.sqrt(naive_mse)
    print(f"Naive forecast - MSE: {naive_mse:.4f}, RMSE: {naive_rmse:.4f}")

    # 7. Prepare the `DataLoader` objects:
    custom_set_seed(42)
    valid_ind = len(X) - valid_size
    X_tensor = torch.from_numpy(X).float().reshape(X.shape[0], X.shape[1], 1)
    y_tensor = torch.from_numpy(y).float().reshape(X.shape[0], 1)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_dataset = Subset(dataset, list(range(valid_ind)))
    valid_dataset = Subset(dataset, list(range(valid_ind, len(X))))

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)

    # Check the size of the datasets:
    print(
        f"Size of datasets - training: {len(train_loader.dataset)} | validation: {len(valid_loader.dataset)}"
    )

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, n_layers, output_size):
            super(RNN, self).__init__()
            self.rnn = nn.RNN(
                input_size, hidden_size, n_layers, batch_first=True, nonlinearity="relu"
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            output, _ = self.rnn(x)
            output = self.fc(output[:, -1, :])
            return output

    model = RNN(input_size=1, hidden_size=6, n_layers=1, output_size=1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    PRINT_EVERY = 10
    train_losses, valid_losses = [], []

    for epoch in range(N_EPOCHS):
        running_loss_train = 0
        running_loss_valid = 0
        model.train()

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_hat = model(x_batch)
            loss = torch.sqrt(loss_fn(y_batch, y_hat))
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item() * x_batch.size(0)
        epoch_loss_train = running_loss_train / len(train_loader.dataset)
        train_losses.append(epoch_loss_train)

        with torch.no_grad():
            model.eval()
            for x_val, y_val in valid_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_hat = model(x_val)
                loss = torch.sqrt(loss_fn(y_val, y_hat))
                running_loss_valid += loss.item() * x_val.size(0)
            epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)
            if epoch > 0 and epoch_loss_valid < min(valid_losses):
                best_epoch = epoch
                torch.save(model.state_dict(), "data/rnn_checkpoint.pth")
            valid_losses.append(epoch_loss_valid)
        if epoch % PRINT_EVERY == 0:
            print(
                f"<{epoch}> - Train. loss: {epoch_loss_train:.4f} \t Valid. loss: {epoch_loss_valid:.4f}"
            )
    print(f"Lowest loss recorded in epoch: {best_epoch}")

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots()
    ax.plot(train_losses, color="blue", label="Training loss")
    ax.plot(valid_losses, color="red", label="Validation loss")
    ax.set(title="Loss over epochs", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/ch10_im14.png")

    # 12. Load the best model (with the lowest validation loss):
    state_dict = torch.load("data/rnn_checkpoint.pth")
    model.load_state_dict(state_dict)

    # 13. Obtain the predictions:
    y_pred = []
    with torch.no_grad():
        model.eval()
        for x_val, y_val in valid_loader:
            x_val = x_val.to(device)
            y_hat = model(x_val)
            y_pred.append(y_hat)
    y_pred = torch.cat(y_pred).cpu().numpy()
    y_pred = minmax.inverse_transform(y_pred).flatten()

    # 14. Evaluate the predictions:
    rnn_mse = mean_squared_error(y_valid, y_pred)
    rnn_rmse = np.sqrt(rnn_mse)
    print(f"RNN's forecast - MSE: {rnn_mse:.4f}, RMSE: {rnn_rmse:.4f}")

    fig, ax = plt.subplots()
    ax.plot(y_valid, color="blue", label="Actual")
    ax.plot(y_pred, color="red", label="RNN")
    ax.plot(naive_pred, color="green", label="Naïve")
    ax.set(title="RNN's Forecasts", xlabel="Time", ylabel="Price ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/ch10_im15.png")
