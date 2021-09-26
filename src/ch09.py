import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from eli5.sklearn import PermutationImportance
from imblearn.ensemble import BalancedRandomForestClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pandas.io.json import json_normalize
from hyperopt.pyll.stochastic import sample

from icecream import ic

plt.style.use("seaborn")
sns.set_palette("cubehelix")
plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 300
warnings.simplefilter(action="ignore", category=FutureWarning)


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


def advanced_classifiers(nsearch=100):
    df = pd.read_csv("data/credit_card_default.csv", index_col=0, na_values="")
    X = df.copy()
    y = X.pop("default_payment_next_month")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    num_features = X_train.select_dtypes(include="number").columns.to_list()
    cat_features = X_train.select_dtypes(include="object").columns.to_list()
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_list = [list(X_train[column].dropna().unique()) for column in cat_features]
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=cat_list, sparse=False, handle_unknown="error", drop="first"
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", num_pipeline, num_features),
            ("categorical", cat_pipeline, cat_features),
        ],
        remainder="drop",
    )
    tree_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )

    LABELS = ["No Default", "Default"]
    tree_pipeline.fit(X_train, y_train)
    tree_perf = performance_evaluation_report(
        tree_pipeline, X_test, y_test, labels=LABELS, show_plot=True, show_pr_curve=True
    )
    ic(tree_perf)
    tree_classifier = tree_pipeline.named_steps["classifier"]
    ic(tree_classifier.tree_.max_depth)

    rf = RandomForestClassifier(random_state=42)
    rf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf)])

    rf_pipeline.fit(X_train, y_train)
    rf_perf = performance_evaluation_report(
        rf_pipeline, X_test, y_test, labels=LABELS, show_plot=True, show_pr_curve=True
    )
    plt.savefig("images/ch9_im1.png", dpi=300)

    gbt = GradientBoostingClassifier(random_state=42)
    gbt_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", gbt)])
    gbt_pipeline.fit(X_train, y_train)
    gbt_perf = performance_evaluation_report(
        gbt_pipeline, X_test, y_test, labels=LABELS, show_plot=True, show_pr_curve=True
    )
    plt.savefig("images/ch9_im2.png", dpi=300)

    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss")
    xgb_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", xgb)])
    xgb_pipeline.fit(X_train, y_train)
    xgb_perf = performance_evaluation_report(
        xgb_pipeline, X_test, y_test, labels=LABELS, show_plot=True, show_pr_curve=True
    )
    plt.savefig("images/ch9_im3.png", dpi=300)

    lgbm = LGBMClassifier(random_state=42, num_leaves=64)
    lgbm_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", lgbm)])
    lgbm_pipeline.fit(X_train, y_train)
    lgbm_perf = performance_evaluation_report(
        lgbm_pipeline, X_test, y_test, labels=LABELS, show_plot=True, show_pr_curve=True
    )
    plt.savefig("images/ch9_im4.png", dpi=300)
    plt.close()

    # Below we go over the most important hyperparameters of the considered models and show a possible
    # way of tuning them using Randomized Search. With more complex models, the training time is significantly
    # longer than with the basic Decision Tree, so we need to find a balance between the time we want to spend
    # on tuning the hyperparameters and the expected results. Also, bear in mind that changing the values of
    # some parameters (such as learning rate or the number of estimators) can itself influence the training
    # time of the models.
    # To have the results in a reasonable amount of time, we used the Randomized Search with 100 different sets
    # of hyperparameters for each model (the number of actually fitted models is higher due to cross-validation).
    # Just as in the recipe *Grid Search and Cross-Validation*, we used recall as the criterion for selecting
    # the best model. Additionally, we used the scikit-learn compatible APIs of XGBoost and LightGBM to make the
    # process as easy to follow as possible. For a complete list of hyperparameters and their meaning,
    # please refer to corresponding documentations.

    N_SEARCHES = nsearch
    k_fold = StratifiedKFold(5, shuffle=True, random_state=42)

    # **Random Forest**
    # When tuning the Random Forest classifier, we look at the following hyperparameters
    # (there are more available for tuning):
    # * `n_estimators` - the number of decision trees in a forest. The goal is to find a balance between improved
    #    accuracy and computational cost.
    # * `max_features` - the maximum number of features considered for splitting a node. The default is the square
    #    root of the number of features. When None, all features are considered.
    # * `max_depth` - the maximum number of levels in each decision tree
    # * `min_samples_split` - the minimum number of observations required to split each node. When set to high it
    #    may cause underfitting, as the trees will not split enough times.
    # * `min_samples_leaf` - the minimum number of data points allowed in a leaf. Too small a value might cause
    #    overfitting, while large values might prevent the tree from growing and cause underfitting.
    # * `bootstrap` - whether to use bootstrapping for each tree in the forest

    rf_param_grid = {
        "classifier__n_estimators": np.linspace(100, 1000, 10, dtype=int),
        "classifier__max_features": ["log2", "sqrt", None],
        "classifier__max_depth": np.arange(3, 11, 1, dtype=int),
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": np.arange(1, 51, 2, dtype=int),
        "classifier__bootstrap": [True, False],
    }
    # And use the randomized search to tune the classifier:
    rf_rs = RandomizedSearchCV(
        rf_pipeline,
        rf_param_grid,
        scoring="recall",
        cv=k_fold,
        n_jobs=-1,
        verbose=1,
        n_iter=N_SEARCHES,
        random_state=42,
    )
    rf_rs.fit(X_train, y_train)
    print(f"Best parameters: {rf_rs.best_params_}")
    print(f"Recall (Training set): {rf_rs.best_score_:.4f}")
    print(f"Recall (Test set): {metrics.recall_score(y_test, rf_rs.predict(X_test)):.4f}")

    rf_rs_perf = performance_evaluation_report(
        rf_rs, X_test, y_test, labels=LABELS, show_plot=True, show_pr_curve=True
    )

    # **Gradient Boosted Trees**
    # As Gradient Boosted Trees are also an ensemble method built on top of decision trees,
    # a lot of the parameters are the same as in the case of the Random Forest. The new one is
    # the learning rate, which is used in the gradient descent algorithm to control the rate of
    # descent towards the minimum of the loss function. When tuning the tree manually, we should
    # consider this hyperparameter together with the number of estimators, as reducing the learning
    # rate (the learning is slower), while increasing the number of estimators can increase
    # the computation time significantly.
    gbt_param_grid = {
        "classifier__n_estimators": np.linspace(100, 1000, 10, dtype=int),
        "classifier__learning_rate": np.arange(0.05, 0.31, 0.05),
        "classifier__max_depth": np.arange(3, 11, 1, dtype=int),
        "classifier__min_samples_split": np.linspace(0.1, 0.5, 12),
        "classifier__min_samples_leaf": np.arange(1, 51, 2, dtype=int),
        "classifier__max_features": ["log2", "sqrt", None],
    }
    gbt_rs = RandomizedSearchCV(
        gbt_pipeline,
        gbt_param_grid,
        scoring="recall",
        cv=k_fold,
        n_jobs=-1,
        verbose=1,
        n_iter=N_SEARCHES,
        random_state=42,
    )
    gbt_rs.fit(X_train, y_train)
    print(f"Best parameters: {gbt_rs.best_params_}")
    print(f"Recall (Training set): {gbt_rs.best_score_:.4f}")
    print(f"Recall (Test set): {metrics.recall_score(y_test, gbt_rs.predict(X_test)):.4f}")
    gbt_rs_perf = performance_evaluation_report(
        gbt_rs, X_test, y_test, labels=LABELS, show_plot=True, show_pr_curve=True
    )
    # **XGBoost**
    # The scikit-learn API of XGBoost makes sure that the hyperparameters are named similarly
    # to their equivalents other scikit-learn's classifiers. So the XGBoost native eta hyperparameter
    # is called learning_rate in scikit-learn's API.
    # The new hyperparameters we consider for this example are:
    # * `min_child_weight` - indicates the minimum sum of weights of all observations required in a child.
    #    This hyperparameter is used for controlling overfitting. Cross-validation should be used for tuning.
    # * `colsample_bytree` - indicates the fraction of columns to be randomly sampled for each tree.
    xgb_param_grid = {
        "classifier__n_estimators": np.linspace(100, 1000, 10, dtype=int),
        "classifier__learning_rate": np.arange(0.05, 0.31, 0.05),
        "classifier__max_depth": np.arange(3, 11, 1, dtype=int),
        "classifier__min_child_weight": np.arange(1, 8, 1, dtype=int),
        "classifier__colsample_bytree": np.linspace(0.3, 1, 7),
    }
    # For defining ranges of parameters that are restricted (such as colsample_bytree which cannot be higher
    # than 1.0) it is better to use `np.linspace` rather than `np.arange`, because the latter allows for some
    # inconsistencies when the step is defined as floating-point. For example, the last value might be 1.0000000002,
    # which then causes an error while training the classifier.
    xgb_rs = RandomizedSearchCV(
        xgb_pipeline,
        xgb_param_grid,
        scoring="recall",
        cv=k_fold,
        n_jobs=-1,
        verbose=1,
        n_iter=N_SEARCHES,
        random_state=42,
    )
    xgb_rs.fit(X_train, y_train)
    print(f"Best parameters: {xgb_rs.best_params_}")
    print(f"Recall (Training set): {xgb_rs.best_score_:.4f}")
    print(f"Recall (Test set): {metrics.recall_score(y_test, xgb_rs.predict(X_test)):.4f}")
    xgb_rs_perf = performance_evaluation_report(
        xgb_rs, X_test, y_test, labels=LABELS, show_plot=True, show_pr_curve=True
    )

    # **LightGBM**
    lgbm_param_grid = {
        "classifier__n_estimators": np.linspace(100, 1000, 10, dtype=int),
        "classifier__learning_rate": np.arange(0.05, 0.31, 0.05),
        "classifier__max_depth": np.arange(3, 11, 1, dtype=int),
        "classifier__colsample_bytree": np.linspace(0.3, 1, 7),
    }
    lgbm_rs = RandomizedSearchCV(
        lgbm_pipeline,
        lgbm_param_grid,
        scoring="recall",
        cv=k_fold,
        n_jobs=-1,
        verbose=1,
        n_iter=N_SEARCHES,
        random_state=42,
    )
    lgbm_rs.fit(X_train, y_train)
    print(f"Best parameters: {lgbm_rs.best_params_}")
    print(f"Recall (Training set): {lgbm_rs.best_score_:.4f}")
    print(f"Recall (Test set): {metrics.recall_score(y_test, lgbm_rs.predict(X_test)):.4f}")
    lgbm_rs_perf = performance_evaluation_report(
        lgbm_rs, X_test, y_test, labels=LABELS, show_plot=True, show_pr_curve=True
    )

    results_dict = {
        "decision_tree_baseline": tree_perf,
        "random_forest": rf_perf,
        "random_forest_rs": rf_rs_perf,
        "gradient_boosted_trees": gbt_perf,
        "gradient_boosted_trees_rs": gbt_rs_perf,
        "xgboost": xgb_perf,
        "xgboost_rs": xgb_rs_perf,
        "light_gbm": lgbm_perf,
        "light_gbm_rs": lgbm_rs_perf,
    }

    results_comparison = pd.DataFrame(results_dict).T
    results_comparison.to_csv("data/results_comparison.csv")
    ic(results_comparison)

    ## Investigating the feature importance
    # in case we have the fitted grid search object `rf_rs`, we extract the best pipeline
    # rf_pipeline = rf_rs.best_estimator_

    rf_classifier = rf_pipeline.named_steps["classifier"]
    preprocessor = rf_pipeline.named_steps["preprocessor"]

    # in case we want to manually assign hyperparameters based on previous grid search
    # best_parameters =  {'n_estimators': 400, 'min_samples_split': 2,
    #                     'min_samples_leaf': 49, 'max_features': None,
    #                     'max_depth': 20, 'bootstrap': True, 'random_state': 42}
    # rf_classifier = rf_classifier.set_params(**best_parameters)

    feat_names = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["onehot"]
        .get_feature_names(input_features=cat_features)
    )
    feat_names = np.r_[num_features, feat_names]

    X_train_preprocessed = pd.DataFrame(preprocessor.transform(X_train), columns=feat_names)

    rf_feat_imp = pd.DataFrame(
        rf_classifier.feature_importances_, index=feat_names, columns=["mdi"]
    )
    rf_feat_imp = rf_feat_imp.sort_values("mdi", ascending=False)
    rf_feat_imp["cumul_importance_mdi"] = np.cumsum(rf_feat_imp.mdi)

    def plot_most_important_features(feat_imp, method="MDI", n_features=10, bottom=False):
        """
        Function for plotting the top/bottom x features in terms of their importance.
        Parameters
        ----------
        feat_imp : pd.Series
            A pd.Series with calculated feature importances
        method : str
            A string representing the method of calculating the importances.
            Used for the title of the plot.
        n_features : int
            Number of top/bottom features to plot
        bottom : boolean
            Indicates if the plot should contain the bottom feature importances.
        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            Ax cointaining the plot
        """

        if bottom:
            indicator = "Bottom"
            feat_imp = feat_imp.sort_values(ascending=True)
        else:
            indicator = "Top"
            feat_imp = feat_imp.sort_values(ascending=False)
        ax = feat_imp.head(n_features).plot.barh()
        ax.invert_yaxis()
        ax.set(
            title=f"Feature importance - {method} ({indicator} {n_features})",
            xlabel="Importance",
            ylabel="Feature",
        )
        return ax

    plot_most_important_features(rf_feat_imp.mdi, method="MDI")
    plt.savefig("images/ch9_im7.png", dpi=300, bbox_inches="tight")

    x_values = range(len(feat_names))
    fig, ax = plt.subplots()
    ax.plot(x_values, rf_feat_imp.cumul_importance_mdi, "b-")
    ax.hlines(y=0.95, xmin=0, xmax=len(x_values), color="g", linestyles="dashed")
    ax.set(title="Cumulative Importances", xlabel="Variable", ylabel="Importance")
    plt.savefig("images/ch9_im8.png", dpi=300, bbox_inches="tight")
    print(
        f"Top 10 features account for {100 * rf_feat_imp.head(10).mdi.sum():.2f}% of the total importance."
    )
    print(
        f"Top {rf_feat_imp[rf_feat_imp.cumul_importance_mdi <= 0.95].shape[0]} features account for 95% of importance."
    )

    perm = PermutationImportance(rf_classifier, n_iter=25, random_state=42)
    perm.fit(X_train_preprocessed, y_train)
    rf_feat_imp["permutation"] = perm.feature_importances_

    plot_most_important_features(rf_feat_imp.permutation, method="Permutation")
    plt.savefig("images/ch9_im9.png", dpi=300, bbox_inches="tight")

    def drop_col_feat_imp(model, X, y, random_state=42):
        """
        Function for calculating the drop column feature importance.
        Parameters
        ----------
        model : scikit-learn's model
            Object representing the estimator with selected hyperparameters.
        X : pd.DataFrame
            Features for training the model
        y : pd.Series
            The target
        random_state : int
            Random state for reproducibility
        Returns
        -------
        importances : list
            List containing the calculated feature importances in the order of appearing in X
        """

        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X, y)
        benchmark_score = model_clone.score(X, y)
        importances = []
        for col in X.columns:
            model_clone = clone(model)
            model_clone.random_state = random_state
            model_clone.fit(X.drop(col, axis=1), y)
            drop_col_score = model_clone.score(X.drop(col, axis=1), y)
            importances.append(benchmark_score - drop_col_score)
        return importances

    rf_feat_imp["drop_column"] = drop_col_feat_imp(
        rf_classifier, X_train_preprocessed, y_train, random_state=42
    )
    plot_most_important_features(rf_feat_imp.drop_column, method="Drop column")
    plt.savefig("images/ch9_im10.png", dpi=300, bbox_inches="tight")

    plot_most_important_features(rf_feat_imp.drop_column, method="Drop column", bottom=True)
    plt.savefig("images/ch9_im11.png", dpi=300, bbox_inches="tight")


def stacking_improved():
    df = pd.read_csv("data/credit_card_fraud.csv")
    X = df.copy()
    y = X.pop("Class")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    k_fold = StratifiedKFold(5, shuffle=True, random_state=42)
    clf_list = [
        ("dec_tree", DecisionTreeClassifier(random_state=42)),
        ("log_reg", LogisticRegression()),
        ("knn", KNeighborsClassifier()),
        ("naive_bayes", GaussianNB()),
    ]

    for model_tuple in clf_list:
        model = model_tuple[1]
        if "random_state" in model.get_params().keys():
            model.set_params(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        recall = metrics.recall_score(y_pred, y_test)
        print(f"{model_tuple[0]}'s recall score: {recall:.4f}")

    lr = LogisticRegression()
    stack_clf = StackingClassifier(clf_list, final_estimator=lr, cv=k_fold, n_jobs=-1)
    stack_clf.fit(X_train, y_train)
    y_pred = stack_clf.predict(X_test)
    recall = metrics.recall_score(y_pred, y_test)
    print(f"The stacked ensemble's recall score: {recall:.4f}")


## Investigating different approaches to handling imbalanced data
def different_approaches():
    df = pd.read_csv("data/credit_card_fraud.csv")
    X = df.copy()
    y = X.pop("Class")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    ic(y.value_counts(normalize=True))

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_perf = performance_evaluation_report(rf, X_test, y_test, show_plot=True, show_pr_curve=True)
    ic(rf_perf)

    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    print(f"The new class proportions are: {dict(Counter(y_rus))}")
    rf.fit(X_rus, y_rus)
    rf_rus_perf = performance_evaluation_report(
        rf, X_test, y_test, show_plot=True, show_pr_curve=True
    )
    ic(rf_rus_perf)

    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X_train, y_train)
    print(f"The new class proportions are: {dict(Counter(y_ros))}")
    rf.fit(X_ros, y_ros)
    rf_ros_perf = performance_evaluation_report(
        rf, X_test, y_test, show_plot=True, show_pr_curve=True
    )
    ic(rf_ros_perf)

    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print(f"The new class proportions are: {dict(Counter(y_smote))}")
    rf.fit(X_smote, y_smote)
    rf_smote_perf = performance_evaluation_report(
        rf, X_test, y_test, show_plot=True, show_pr_curve=True
    )
    ic(rf_smote_perf)

    X_adasyn, y_adasyn = ADASYN(random_state=42).fit_resample(X_train, y_train)
    print(f"The new class proportions are: {dict(Counter(y_adasyn))}")
    rf.fit(X_adasyn, y_adasyn)
    rf_adasyn_perf = performance_evaluation_report(
        rf, X_test, y_test, show_plot=True, show_pr_curve=True
    )
    ic(rf_adasyn_perf)

    rf_cw = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    rf_cw.fit(X_train, y_train)
    rf_cw_perf = performance_evaluation_report(
        rf_cw, X_test, y_test, show_plot=True, show_pr_curve=True
    )
    ic(rf_cw_perf)

    balanced_rf = BalancedRandomForestClassifier(random_state=42)
    balanced_rf.fit(X_train, y_train)
    balanced_rf_perf = performance_evaluation_report(
        balanced_rf, X_test, y_test, show_plot=True, show_pr_curve=True
    )

    balanced_rf_cw = BalancedRandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    balanced_rf_cw.fit(X_train, y_train)
    balanced_rf_cw_perf = performance_evaluation_report(
        balanced_rf_cw, X_test, y_test, show_plot=True, show_pr_curve=True
    )

    performance_results = {
        "random_forest": rf_perf,
        "undersampled rf": rf_rus_perf,
        "oversampled_rf": rf_ros_perf,
        "smote": rf_smote_perf,
        "adasyn": rf_adasyn_perf,
        "random_forest_cw": rf_cw_perf,
        "balanced_random_forest": balanced_rf_perf,
        "balanced_random_forest_cw": balanced_rf_cw_perf,
    }
    results = pd.DataFrame(performance_results).T
    results.to_csv("data/performance_results.csv")
    ic(results)


def bayesian_optimization(evals=200):
    df = pd.read_csv("data/credit_card_fraud.csv")
    X = df.copy()
    y = X.pop("Class")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    N_FOLDS = 5
    MAX_EVALS = evals

    def objective(params, n_folds=N_FOLDS, random_state=42):
        model = LGBMClassifier(**params, num_leaves=64)
        model.set_params(random_state=random_state)
        k_fold = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)
        metrics = cross_val_score(model, X_train, y_train, cv=k_fold, scoring="recall")
        loss = -1 * metrics.mean()
        return {"loss": loss, "params": params, "status": STATUS_OK}

    lgbm_param_grid = {
        "boosting_type": hp.choice("boosting_type", ["gbdt", "dart", "goss"]),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "n_estimators": hp.choice("n_estimators", [10, 50, 100, 300, 750, 1000]),
        "is_unbalance": hp.choice("is_unbalance", [True, False]),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
        "learning_rate": hp.uniform("learning_rate", 0.05, 0.3),
    }

    trials = Trials()
    best_set = fmin(
        fn=objective, space=lgbm_param_grid, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials
    )

    # load if already finished the search
    # best_set = pickle.load(open("data/best_set.p", "rb"))
    ic(best_set)

    boosting_type = {0: "gbdt", 1: "dart", 2: "goss"}
    max_depth = {0: -1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}
    n_estimators = {0: 10, 1: 50, 2: 100, 3: 300, 4: 750, 5: 1000}
    is_unbalance = {0: True, 1: False}

    lgbm_param_grid = {
        "boosting_type": hp.choice("boosting_type", ["gbdt", "dart", "goss"]),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "n_estimators": hp.choice("n_estimators", [10, 50, 100, 300, 750, 1000]),
        "is_unbalance": hp.choice("is_unbalance", [True, False]),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
        "learning_rate": hp.uniform("learning_rate", 0.05, 0.3),
    }

    best_lgbm = LGBMClassifier(
        boosting_type=boosting_type[best_set["boosting_type"]],
        max_depth=max_depth[best_set["max_depth"]],
        n_estimators=n_estimators[best_set["n_estimators"]],
        is_unbalance=is_unbalance[best_set["is_unbalance"]],
        colsample_bytree=best_set["colsample_bytree"],
        learning_rate=best_set["learning_rate"],
        num_leaves=64,
    )
    best_lgbm.fit(X_train, y_train)

    _ = performance_evaluation_report(best_lgbm, X_test, y_test, show_plot=True, show_pr_curve=True)
    plt.savefig("images/ch9_im13.png", dpi=300)

    trials = pickle.load(open("data/trials_final.p", "rb"))
    results_df = pd.DataFrame(trials.results)
    params_df = json_normalize(results_df["params"])
    results_df = pd.concat([results_df.drop("params", axis=1), params_df], axis=1)
    results_df["iteration"] = np.arange(len(results_df)) + 1
    results_df.sort_values("loss")

    colsample_bytree_dist = []
    for _ in range(10000):
        x = sample(lgbm_param_grid["colsample_bytree"])
        colsample_bytree_dist.append(x)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.kdeplot(colsample_bytree_dist, label="Sampling Distribution", ax=ax[0])
    sns.kdeplot(results_df["colsample_bytree"], label="Bayesian Optimization", ax=ax[0])
    ax[0].set(title="Distribution of colsample_bytree", xlabel="Value", ylabel="Density")
    ax[0].legend()
    sns.regplot("iteration", "colsample_bytree", data=results_df, ax=ax[1])
    ax[1].set(title="colsample_bytree over Iterations", xlabel="Iteration", ylabel="Value")
    plt.savefig("images/ch9_im14.png", dpi=300)

    results_df["n_estimators"].value_counts().plot.bar(title="# of Estimators, Distribution")
    plt.savefig("images/ch9_im15.png", dpi=300)

    fig, ax = plt.subplots()
    ax.plot(results_df.iteration, results_df.loss, "o")
    ax.set(title="TPE Sequence of Losses", xlabel="Iteration", ylabel="Loss")
    plt.savefig("images/ch9_im16.png", dpi=300)


if __name__ == "__main__":
    start = time.time()
    # advanced_classifiers(nsearch=2)
    # stacking_improved()
    # different_approaches()
    bayesian_optimization(evals=2)
    ic(f"elapsed time : {time.time()-start}")
