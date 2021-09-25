import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import plotly.express as px
import plotly.io as pio
import pandas_profiling
import missingno
import category_encoders as ce
import pydotplus

from io import StringIO
from ipywidgets import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    RandomizedSearchCV,
    cross_validate,
    StratifiedKFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics

from datetime import date, datetime
from dotenv import load_dotenv
from icecream import ic

plt.style.use("seaborn")
sns.set_palette("cubehelix")
plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 300
warnings.simplefilter(action="ignore", category=FutureWarning)

# Identifying Credit Card Default with Machine Learning
# All the recipes use `scikit-learn` version `0.21` (unless specified otherwise).
# From `0.22`, the default settings of selected estimators are changed. For example,
# in the case of the `RandomForestClassifier`, the default setting of `n_estimators`
# was changed from 10 to 100. This will cause discrepancies with the results presented in the book.

## BONUS: Getting the data and preparing for book
# This is a part not covered in the book. We download the considered dataset
# from the website of the [UC Irvine Machine Learning Repository]
# (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).
# The dataset originally does not contain missing values and the categorical variables are already
# encoded as numbers. To show the entire pipeline of working and preparing potentially messy data,
# we apply some transformations:
# * we encoded the gender, education and marital status related variables to strings
# * we introduced missing values to some observations (0.5% of the entire sample,
# selected randomly per column - the total percentage of rows with at least one missing value will be higher)
# * some observed values for features such as level of education, payment status, etc. are outside of the range
# of possible categories defined by the authors. As this problem affects many observations, we encode new,
# undescribed categories as either 'Others' (when there was already such a category) or 'Unknown'
# (in the case of payment status).
# The reason for selecting only a small fraction of values to be missing is that we do not want
# to significantly change the underlying structure/patterns in the data.
# downloading the data
# !wget https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls


def prepare_csv_missing():
    df = pd.read_excel("data/default of credit card clients.xlsx", skiprows=1, index_col=0)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    ic(df.head())

    months = ["sep", "aug", "jul", "jun", "may", "apr"]
    variables = ["payment_status", "bill_statement", "previous_payment"]
    new_column_names = [x + "_" + y for x in variables for y in months]
    rename_dict = {x: y for x, y in zip(df.loc[:, "pay_0":"pay_amt6"].columns, new_column_names)}
    df.rename(columns=rename_dict, inplace=True)
    ic(df.sex.value_counts())
    ic(df.education.value_counts(sort=True).index.sort_values(ascending=True))
    ic(df.marriage.value_counts())
    ic(df.age.describe(include="all"))

    # creating dicts to map number to strings
    gender_dict = {1: "Male", 2: "Female"}
    education_dict = {
        0: "Others",
        1: "Graduate school",
        2: "University",
        3: "High school",
        4: "Others",
        5: "Others",
        6: "Others",
    }
    marital_status_dict = {0: "Others", 1: "Married", 2: "Single", 3: "Others"}
    payment_status = {
        -2: "Unknown",
        -1: "Payed duly",
        0: "Unknown",
        1: "Payment delayed 1 month",
        2: "Payment delayed 2 months",
        3: "Payment delayed 3 months",
        4: "Payment delayed 4 months",
        5: "Payment delayed 5 months",
        6: "Payment delayed 6 months",
        7: "Payment delayed 7 months",
        8: "Payment delayed 8 months",
        9: "Payment delayed >= 9 months",
    }

    # # map numbers to strings
    df.sex = df.sex.map(gender_dict)
    df.education = df.education.map(education_dict)
    df.marriage = df.marriage.map(marital_status_dict)
    for column in [x for x in df.columns if ("status" in x)]:
        df[column] = df[column].map(payment_status)

    # define the ratio of missing values to introduce
    RATIO_MISSING = 0.005
    random_state = np.random.RandomState(42)
    for column in ["sex", "education", "marriage", "age"]:
        df.loc[df.sample(frac=RATIO_MISSING, random_state=random_state).index, column] = ""
    df.reset_index(drop=True, inplace=True)
    df.to_csv("data/credit_card_default.csv")


def performance_evaluation_report(
    model, X_test, y_test, show_plot=False, labels=None, show_pr_curve=False
):
    """
    Function for creating a performance report of a classification model.
    Parameters
    ----------
    model : scikit-learn estimator
        A fitted estimator for classification problems.
    X_test : pd.DataFrame
        DataFrame with features matching y_test
    y_test : array/pd.Series
        Target of a classification problem.
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
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    cm = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_prob)
    pr_auc = metrics.auc(recall, precision)

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
            ax[2].plot(recall, precision, label=f"PR-AUC = {pr_auc:.2f}")
            ax[2].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
            ax[2].legend()

    #         print('#######################')
    #         print('Evaluation metrics ####')
    #         print('#######################')
    #         print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}')
    #         print(f'Precision: {metrics.precision_score(y_test, y_pred):.4f}')
    #         print(f'Recall (Sensitivity): {metrics.recall_score(y_test, y_pred):.4f}')
    #         print(f'Specificity: {(tn / (tn + fp)):.4f}')
    #         print(f'F1-Score: {metrics.f1_score(y_test, y_pred):.4f}')
    #         print(f"Cohen's Kappa: {metrics.cohen_kappa_score(y_test, y_pred):.4f}")

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


def exploratory_data_analysis(df):
    print(f"The DataFrame has {len(df)} rows and {df.shape[1]} columns.")

    def get_df_memory_usage(df, top_columns=5):
        """
        Function for quick analysis of a pandas DataFrame's memory usage.
        It prints the top `top_columns` columns in terms of memory usage
        and the total usage of the DataFrame.
        Parameters
        ------------
        df : pd.DataFrame
            DataFrame to be inspected
        top_columns : int
            Number of top columns (in terms of memory used) to display
        """
        print("Memory usage ----")
        memory_per_column = df.memory_usage(deep=True) / 1024 ** 2
        print(f"Top {top_columns} columns by memory (MB):")
        print(memory_per_column.sort_values(ascending=False).head(top_columns))
        print(f"Total size: {memory_per_column.sum():.4f} MB")

    get_df_memory_usage(df, 5)

    df_cat = df.copy()
    object_columns = df_cat.select_dtypes(include="object").columns
    df_cat[object_columns] = df_cat[object_columns].astype("category")
    get_df_memory_usage(df_cat)

    column_dtypes = {
        "education": "category",
        "marriage": "category",
        "sex": "category",
        "payment_status_sep": "category",
        "payment_status_aug": "category",
        "payment_status_jul": "category",
        "payment_status_jun": "category",
        "payment_status_may": "category",
        "payment_status_apr": "category",
    }
    df_cat2 = pd.read_csv(
        "data/credit_card_default.csv", index_col=0, na_values="", dtype=column_dtypes
    )

    get_df_memory_usage(df_cat2)

    ic(df_cat.equals(df_cat2))
    ic(df.describe(include="number").transpose().round(2))
    ic(df.describe(include="object").T)
    ic(df.isna().any().any())
    ic(df.isna().sum())

    fig, ax = plt.subplots()
    sns.distplot(
        df.loc[df.sex == "Male", "age"].dropna(),
        hist=False,
        color="green",
        kde_kws={"shade": True},
        ax=ax,
        label="Male",
    )
    sns.distplot(
        df.loc[df.sex == "Female", "age"].dropna(),
        hist=False,
        color="blue",
        kde_kws={"shade": True},
        ax=ax,
        label="Female",
    )
    ax.set_title("Distribution of age")
    ax.legend(title="Gender:")
    plt.tight_layout()
    plt.savefig("images/ch8_im5.png")

    # As mentioned in the text, we can create a histogram (together with the KDE), by calling:
    ax = sns.distplot(df.age.dropna(), hist=True)
    ax.set_title("Distribution of age")
    plt.tight_layout()
    plt.savefig("images/ch8_im51.png", bbox_inches="tight")

    # We noticed some spikes appearing every ~10 years and the reason for this is the binning.
    # Below, we created the same histogram using `sns.countplot` and `plotly_express`. By doing so,
    # each value of age has a separate bin and we can inspect the plot in detail.
    # There are no such spikes in the following plots:

    plot_ = sns.countplot(x=df.age.dropna(), color="blue")
    for ind, label in enumerate(plot_.get_xticklabels()):
        if int(float(label.get_text())) % 10 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    px.histogram(df, x="age", title="Distribution of age")
    plt.savefig("images/ch8_im52.png", bbox_inches="tight")

    pair_plot = sns.pairplot(df[["age", "limit_bal", "previous_payment_sep"]])
    pair_plot.fig.suptitle("Pairplot of selected variables", y=1.05)
    plt.savefig("images/ch8_im6.png", bbox_inches="tight")

    # Additionally, we can separate the genders by specifying the `hue` argument:
    pair_plot = sns.pairplot(
        df[["sex", "age", "limit_bal", "previous_payment_sep"]], hue="sex", diag_kind="hist"
    )
    pair_plot.fig.suptitle("Pairplot of selected variables", y=1.05)
    plt.savefig("images/ch8_im6_1.png", bbox_inches="tight")

    def plot_correlation_matrix(corr_mat):
        """
        Function for plotting the correlation heatmap. It masks the irrelevant fields.
        Parameters
        ----------
        corr_mat : pd.DataFrame
            Correlation matrix of the features.
        """
        # temporarily change style
        sns.set(style="white")
        # mask the upper triangle
        mask = np.zeros_like(corr_mat, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        # set up the matplotlib figure
        fig, ax = plt.subplots()
        # set up custom diverging colormap
        cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
        # plot the heatmap
        sns.heatmap(
            corr_mat,
            mask=mask,
            cmap=cmap,
            vmax=0.3,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            ax=ax,
        )
        ax.set_title("Correlation Matrix", fontsize=16)
        # change back to darkgrid style
        sns.set(style="darkgrid")

    corr_mat = df.select_dtypes(include="number").corr()
    plot_correlation_matrix(corr_mat)
    plt.savefig("images/ch8_im7.png", bbox_inches="tight")

    # We can also directly inspect the correlation between the features (numerical) and the target:
    ic(df.select_dtypes(include="number").corr()[["default_payment_next_month"]])

    # 7. Plot the distribution of limit balance for each gender and education level:
    plt.clf()
    ax = sns.violinplot(x="education", y="limit_bal", hue="sex", split=True, data=df)
    ax.set_title("Distribution of limit balance per education level", fontsize=16)
    plt.savefig("images/ch8_im8.png", bbox_inches="tight")

    # The following code plots the same information, without splitting the violin plots.
    ax = sns.violinplot(x="education", y="limit_bal", hue="sex", data=df)
    ax.set_title("Distribution of limit balance per education level", fontsize=16)
    plt.savefig("images/ch8_im81.png", bbox_inches="tight")

    # 8. Investigate the distribution of the target variable per gender and education level:
    ax = sns.countplot("default_payment_next_month", hue="sex", data=df, orient="h")
    ax.set_title("Distribution of the target variable", fontsize=16)
    plt.savefig("images/ch8_im9.png", bbox_inches="tight")

    # 9. Investigate the percentage of defaults per education level:
    ax = (
        df.groupby("education")["default_payment_next_month"]
        .value_counts(normalize=True)
        .unstack()
        .plot(kind="barh", stacked="True")
    )
    ax.set_title("Percentage of default per education level", fontsize=16)
    ax.legend(title="Default", bbox_to_anchor=(1, 1))
    plt.savefig("images/ch8_im10.png", bbox_inches="tight")
    plt.close()

    # profile = df.profile_report()
    # profile.to_file("data/profile.html")


def split_datasets(df):
    X = df.copy()
    y = X.pop("default_payment_next_month")
    ic(df.dtypes)

    ## Splitting the data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # # 3. Split the data into training and test sets without shuffling:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # # 4. Split the data into training and test sets with stratification:
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, stratify=y, random_state=42
    # )
    # # 5. Verify that the ratio of the target is preserved:
    # y_train.value_counts(normalize=True)
    # y_test.value_counts(normalize=True)

    # define the size of the validation and test sets
    VALID_SIZE = 0.1
    TEST_SIZE = 0.2
    # create the initial split - training and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VALID_SIZE + TEST_SIZE), stratify=y, random_state=42
    )

    # calculate the new test size
    NEW_TEST_SIZE = np.around(TEST_SIZE / (VALID_SIZE + TEST_SIZE), 2)
    # create the valid and test sets
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=NEW_TEST_SIZE, stratify=y_temp, random_state=42
    )
    #
    ## Dealing with missing values
    missingno.matrix(X)
    plt.savefig("images/ch8_im12.png")

    # 4. Define columns with missing values per data type:
    NUM_FEATURES = ["age"]
    CAT_FEATURES = ["sex", "education", "marriage"]

    X_train = X_train.copy()
    X_test = X_test.copy()

    # 5. Impute the numerical feature:
    for col in NUM_FEATURES:
        num_imputer = SimpleImputer(strategy="median")
        num_imputer.fit(X_train[[col]])
        X_train.loc[:, col] = num_imputer.transform(X_train[[col]])
        X_test.loc[:, col] = num_imputer.transform(X_test[[col]])

    # alternative method using pandas
    # for feature in NUM_FEATURES:
    #     median_value = X_train[feature].median()
    #     X_train.loc[:, feature].fillna(median_value, inplace=True)
    #     X_test.loc[:, feature].fillna(median_value, inplace=True)

    # 6. Impute the categorical features:
    for col in CAT_FEATURES:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cat_imputer.fit(X_train[[col]])
        X_train.loc[:, col] = cat_imputer.transform(X_train[[col]])
        X_test.loc[:, col] = cat_imputer.transform(X_test[[col]])

    # alternative method using pandas
    for feature in CAT_FEATURES:
        ic(X_train[feature].mode().values[0])
    #     mode_value = X_train[feature].mode().values[0]
    #     X_train.loc[:, feature].fillna(mode_value, inplace=True)
    #     X_test.loc[:, feature].fillna(mode_value, inplace=True)

    # 7. Verify that there are no missing values:
    # X_train.info()
    ic(X_train.isna().any().any())
    ic(X_train.isna().sum())

    ## Encoding categorical variables
    # 2. Use Label Encoder to encode a selected column:
    COL = "education"
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    label_enc = LabelEncoder()
    label_enc.fit(X_train_copy[COL])
    X_train_copy.loc[:, COL] = label_enc.transform(X_train_copy[COL])
    X_test_copy.loc[:, COL] = label_enc.transform(X_test_copy[COL])
    # 3. Select categorical features for one-hot encoding:
    CAT_FEATURES = X_train.select_dtypes(include="object").columns.to_list()
    # 4. Instantiate the One-Hot Encoder object:
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="error", drop="first")
    # 5. Create the column transformer using the one-hot encoder:
    one_hot_transformer = ColumnTransformer(
        [("one_hot", one_hot_encoder, CAT_FEATURES)]
        # ,remainder='passthrough'
    )
    # 6. Fit the transformer:
    one_hot_transformer.fit(X_train)
    # 7. Apply the transformations to both training and test sets:
    col_names = one_hot_transformer.get_feature_names()
    X_train_cat = pd.DataFrame(
        one_hot_transformer.transform(X_train), columns=col_names, index=X_train.index
    )
    X_train_ohe = pd.concat([X_train, X_train_cat], axis=1).drop(CAT_FEATURES, axis=1)

    X_test_cat = pd.DataFrame(
        one_hot_transformer.transform(X_test), columns=col_names, index=X_test.index
    )
    X_test_ohe = pd.concat([X_test, X_test_cat], axis=1).drop(CAT_FEATURES, axis=1)

    #### Using `pandas.get_dummies` for one-hot encoding
    ic(pd.get_dummies(X_train, prefix_sep="_", drop_first=True))

    #### Specifying possible categories for OneHotEncoder
    one_hot_encoder = OneHotEncoder(
        categories=[["Male", "Female", "Unknown"]],
        sparse=False,
        handle_unknown="error",
        drop="first",
    )
    one_hot_transformer = ColumnTransformer([("one_hot", one_hot_encoder, ["sex"])])
    one_hot_transformer.fit(X_train)
    ic(one_hot_transformer.get_feature_names())

    #### Category Encoders library
    one_hot_encoder_ce = ce.OneHotEncoder(use_cat_names=True)
    one_hot_encoder_ce.fit(X_train)
    X_train_ce = one_hot_encoder_ce.transform(X_train)
    ic(X_train_ce.head())

    target_encoder = ce.TargetEncoder(smoothing=0)
    target_encoder.fit(X_train.sex, y_train)
    target_encoder.transform(X_train.sex).head()

    ## Fitting a decision tree classifier
    tree_classifier = DecisionTreeClassifier(random_state=42)
    tree_classifier.fit(X_train_ohe, y_train)
    y_pred = tree_classifier.predict(X_test_ohe)
    ic(y_pred)

    LABELS = ["No Default", "Default"]
    tree_perf = performance_evaluation_report(
        tree_classifier, X_test_ohe, y_test, labels=LABELS, show_plot=True
    )
    plt.savefig("images/ch8_im14.png", bbox_inches="tight")
    ic(tree_perf)

    # 4. Plot the simplified Decision Tree:
    small_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    small_tree.fit(X_train_ohe, y_train)
    tree_dot = StringIO()
    export_graphviz(
        small_tree,
        feature_names=X_train_ohe.columns,
        class_names=LABELS,
        rounded=True,
        out_file=tree_dot,
        proportion=False,
        precision=2,
        filled=True,
    )
    tree_graph = pydotplus.graph_from_dot_data(tree_dot.getvalue())
    tree_graph.set_dpi(300)
    tree_graph.write_png("images/ch8_im15.png")
    # Image(value=tree_graph.create_png())

    y_pred_prob = tree_classifier.predict_proba(X_test_ohe)[:, 1]
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_prob)

    plt.clf()
    ax = plt.subplot()
    ax.plot(recall, precision, label=f"PR-AUC = {metrics.auc(recall, precision):.2f}")
    ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax.legend()
    plt.savefig("images/ch8_im16.png", bbox_inches="tight")


## Implementing scikit-learn's pipelines
def sci_pipelines(df):
    X = df.copy()
    y = X.pop("default_payment_next_month")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    num_features = X_train.select_dtypes(include="number").columns.to_list()
    cat_features = X_train.select_dtypes(include="object").columns.to_list()

    # 4. Define the numerical pipeline:
    cat_list = [list(X_train[col].dropna().unique()) for col in cat_features]
    one_hot_encoder = OneHotEncoder(
        categories=cat_list, sparse=False, handle_unknown="error", drop="first"
    )

    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder),
        ]
    )
    # 6. Define the column transformer object:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", num_pipeline, num_features),
            ("categorical", cat_pipeline, cat_features),
        ],
        remainder="drop",
    )
    # 7. Create the joint pipeline:
    dec_tree = DecisionTreeClassifier(random_state=42)
    tree_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", dec_tree)])
    tree_pipeline.fit(X_train, y_train)
    LABELS = ["No Default", "Default"]
    tree_perf = performance_evaluation_report(
        tree_pipeline, X_test, y_test, labels=LABELS, show_plot=True
    )
    ic(tree_perf)
    plt.savefig("images/ch8_im17.png", bbox_inches="tight")

    class OutlierRemover(BaseEstimator, TransformerMixin):
        def __init__(self, n_std=3):
            self.n_std = n_std
            self.lower_band_ = 0
            self.upper_band_ = 0
            self.n_features_ = 0

        def fit(self, X, y=None):
            if np.isnan(X).any(axis=None):
                raise ValueError("There are missing values in the array! Please remove them.")
            mean_vec = np.mean(X, axis=0)
            std_vec = np.std(X, axis=0)
            self.lower_band_ = mean_vec - self.n_std * std_vec
            self.upper_band_ = mean_vec + self.n_std * std_vec
            self.n_features_ = len(self.upper_band_)
            return self

        def transform(self, X, y=None):
            X_copy = pd.DataFrame(X.copy())
            upper_band = np.repeat(
                self.upper_band_.reshape(self.n_features_, -1), len(X_copy), axis=1
            ).transpose()
            lower_band = np.repeat(
                self.lower_band_.reshape(self.n_features_, -1), len(X_copy), axis=1
            ).transpose()
            X_copy[X_copy >= upper_band] = upper_band
            X_copy[X_copy <= lower_band] = lower_band
            return X_copy.values

    num_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("outliers", OutlierRemover())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", num_pipeline, num_features),
            ("categorical", cat_pipeline, cat_features),
        ],
        remainder="drop",
    )
    dec_tree = DecisionTreeClassifier(random_state=42)
    tree_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", dec_tree)])
    tree_pipeline.fit(X_train, y_train)
    tree_perf = performance_evaluation_report(
        tree_pipeline, X_test, y_test, labels=LABELS, show_plot=True
    )
    ic(tree_perf)
    plt.savefig("images/ch8_im18.png", bbox_inches="tight")

    return tree_pipeline


def tuning_parameters(pipeline):
    X = df.copy()
    y = X.pop("default_payment_next_month")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    ## Tuning hyperparameters using grid search and cross-validation
    k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    cross_val_score(pipeline, X_train, y_train, cv=k_fold)
    cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=k_fold,
        scoring=["accuracy", "precision", "recall", "roc_auc"],
    )
    param_grid = {
        "classifier__criterion": ["entropy", "gini"],
        "classifier__max_depth": range(3, 11),
        "classifier__min_samples_leaf": range(2, 11),
        "preprocessor__numerical__outliers__n_std": [3, 4],
    }

    classifier_gs = GridSearchCV(
        pipeline, param_grid=param_grid, scoring="recall", cv=k_fold, n_jobs=-1, verbose=1
    )
    classifier_gs.fit(X_train, y_train)

    print(f"Best parameters: {classifier_gs.best_params_}")
    print(f"Recall (Training set): {classifier_gs.best_score_:.4f}")
    print(f"Recall (Test set): {metrics.recall_score(y_test, classifier_gs.predict(X_test)):.4f}")

    LABELS = ["No Default", "Default"]
    tree_gs_perf = performance_evaluation_report(
        classifier_gs, X_test, y_test, labels=LABELS, show_plot=True
    )
    ic(tree_gs_perf)
    plt.savefig("images/ch8_im20.png", bbox_inches="tight")

    # classifier_rs = RandomizedSearchCV(
    #     pipeline,
    #     param_grid,
    #     scoring="recall",
    #     cv=k_fold,
    #     n_jobs=-1,
    #     verbose=1,
    #     n_iter=100,
    #     random_state=42,
    # )
    # classifier_rs.fit(X_train, y_train)
    #
    # print(f"Best parameters: {classifier_rs.best_params_}")
    # print(f"Recall (Training set): {classifier_rs.best_score_:.4f}")
    # print(f"Recall (Test set): {metrics.recall_score(y_test, classifier_rs.predict(X_test)):.4f}")
    #
    # tree_rs_perf = performance_evaluation_report(
    #     classifier_rs, X_test, y_test, labels=LABELS, show_plot=True
    # )
    # ic(tree_rs_perf)
    # plt.savefig("images/ch8_im21.png", bbox_inches="tight")
    #
    # param_grid = [
    #     {
    #         "classifier": [LogisticRegression()],
    #         "classifier__penalty": ["l1", "l2"],
    #         "classifier__C": np.logspace(0, 3, 10, 2),
    #         "preprocessor__numerical__outliers__n_std": [3, 4],
    #     },
    #     {
    #         "classifier": [DecisionTreeClassifier(random_state=42)],
    #         "classifier__criterion": ["entropy", "gini"],
    #         "classifier__max_depth": range(3, 11),
    #         "classifier__min_samples_leaf": range(2, 11),
    #         "preprocessor__numerical__outliers__n_std": [3, 4],
    #     },
    # ]
    #
    # classifier_gs_2 = GridSearchCV(
    #     pipeline, param_grid=param_grid, scoring="recall", cv=k_fold, n_jobs=-1, verbose=1
    # )
    # classifier_gs_2.fit(X_train, y_train)
    #
    # print(f"Best parameters: {classifier_gs_2.best_params_}")
    # print(f"Recall (Training set): {classifier_gs_2.best_score_:.4f}")
    # print(f"Recall (Test set): {metrics.recall_score(y_test, classifier_gs_2.predict(X_test)):.4f}")


if __name__ == "__main__":
    df = pd.read_csv("data/credit_card_default.csv", index_col=0, na_values="")

    # prepare_csv_missing()
    # exploratory_data_analysis(df)
    # split_datasets(df)
    pipeline = sci_pipelines(df)
    tuning_parameters(pipeline)
