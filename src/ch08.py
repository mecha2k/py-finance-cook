import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random

from datetime import date, datetime
from dotenv import load_dotenv
from icecream import ic

plt.style.use("seaborn")
sns.set_palette("cubehelix")
plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 150
warnings.simplefilter(action="ignore", category=FutureWarning)

# Identifying Credit Card Default with Machine Learning
# All the recipes use `scikit-learn` version `0.21` (unless specified otherwise).
# From `0.22`, the default settings of selected estimators are changed. For example,
# in the case of the `RandomForestClassifier`, the default setting of `n_estimators`
# was changed from 10 to 100. This will cause discrepancies with the results presented in the book.

# ## BONUS: Getting the data and preparing for book
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

if __name__ == "__main__":
    # loading the data from Excel
    df = pd.read_excel("data/default of credit card clients.xlsx", skiprows=1, index_col=0)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    months = ["sep", "aug", "jul", "jun", "may", "apr"]
    variables = ["payment_status", "bill_statement", "previous_payment"]
    new_column_names = [x + "_" + y for x in variables for y in months]
    rename_dict = {x: y for x, y in zip(df.loc[:, "pay_0":"pay_amt6"].columns, new_column_names)}
    df.rename(columns=rename_dict, inplace=True)

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

    df = pd.read_csv("data/credit_card_default.csv", index_col=0, na_values="")
    print(f"The DataFrame has {len(df)} rows and {df.shape[1]} columns.")
    ic(df.head())

    X = df.copy()
    y = X.pop("default_payment_next_month")
    ic(df.dtypes)

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
    df_cat.equals(df_cat2)

    # # ## Exploratory Data Analysis
    #
    # # ### How to do it...
    #
    # # 1. Import the libraries:
    #
    # # In[15]:
    #
    #
    # import pandas as pd
    # import seaborn as sns
    # import numpy as np
    # import plotly.express as px
    # import plotly.io as pio
    #
    #
    # # 2. Get summary statistics for numeric variables:
    #
    # # In[18]:
    #
    #
    # df.describe().transpose().round(2)
    #
    #
    # # 3. Get summary statistics for categorical variables:
    #
    # # In[19]:
    #
    #
    # df.describe(include='object').transpose()
    #
    #
    # # 4. Plot the distribution of age and split it by gender:
    #
    # # In[20]:
    #
    #
    # fig, ax = plt.subplots()
    # sns.distplot(df.loc[df.sex=='Male', 'age'].dropna(),
    #              hist=False, color='green',
    #              kde_kws={'shade': True},
    #              ax=ax, label='Male')
    # sns.distplot(df.loc[df.sex=='Female', 'age'].dropna(),
    #              hist=False, color='blue',
    #              kde_kws={'shade': True},
    #              ax=ax, label='Female')
    # ax.set_title('Distribution of age')
    # ax.legend(title='Gender:')
    #
    # plt.tight_layout()
    # # plt.savefig('images/ch8_im5.png')
    # plt.show()
    #
    #
    # # As mentioned in the text, we can create a histogram (together with the KDE), by calling:
    #
    # # In[21]:
    #
    #
    # ax = sns.distplot(df.age.dropna(), )
    # ax.set_title('Distribution of age');
    #
    #
    # # We noticed some spikes appearing every ~10 years and the reason for this is the binning. Below, we created the same histogram using `sns.countplot` and `plotly_express`. By doing so, each value of age has a separate bin and we can inspect the plot in detail. There are no such spikes in the following plots:
    #
    # # In[22]:
    #
    #
    # plot_ = sns.countplot(x=df.age.dropna(), color='blue')
    #
    # for ind, label in enumerate(plot_.get_xticklabels()):
    #     if int(float(label.get_text())) % 10 == 0:
    #         label.set_visible(True)
    #     else:
    #         label.set_visible(False)
    #
    #
    # # In[24]:
    #
    #
    # px.histogram(df, x='age', title = 'Distribution of age')
    #
    #
    # # 5. Plot a `pairplot` of selected variables:
    #
    # # In[23]:
    #
    #
    # pair_plot = sns.pairplot(df[['age', 'limit_bal', 'previous_payment_sep']])
    # pair_plot.fig.suptitle('Pairplot of selected variables', y=1.05)
    #
    # plt.tight_layout()
    # # plt.savefig('images/ch8_im6.png', bbox_inches='tight')
    # plt.show()
    #
    #
    # # Additionally, we can separate the genders by specifying the `hue` argument:
    #
    # # In[18]:
    #
    #
    # # pair_plot = sns.pairplot(df[['sex', 'age', 'limit_bal', 'previous_payment_sep']],
    # #                          hue='sex')
    # # pair_plot.fig.suptitle('Pairplot of selected variables', y=1.05);
    #
    #
    # # 6. Define and run a function for plotting the correlation heatmap:
    #
    # # In[25]:
    #
    #
    # def plot_correlation_matrix(corr_mat):
    #     '''
    #     Function for plotting the correlation heatmap. It masks the irrelevant fields.
    #
    #     Parameters
    #     ----------
    #     corr_mat : pd.DataFrame
    #         Correlation matrix of the features.
    #     '''
    #
    #     # temporarily change style
    #     sns.set(style='white')
    #     # mask the upper triangle
    #     mask = np.zeros_like(corr_mat, dtype=np.bool)
    #     mask[np.triu_indices_from(mask)] = True
    #     # set up the matplotlib figure
    #     fig, ax = plt.subplots()
    #     # set up custom diverging colormap
    #     cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
    #     # plot the heatmap
    #     sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=.3, center=0,
    #                 square=True, linewidths=.5,
    #                 cbar_kws={'shrink': .5}, ax=ax)
    #     ax.set_title('Correlation Matrix', fontsize=16)
    #     # change back to darkgrid style
    #     sns.set(style='darkgrid')
    #
    #
    # # In[36]:
    #
    #
    # corr_mat = df.select_dtypes(include='number').corr()
    # plot_correlation_matrix(corr_mat)
    #
    # plt.tight_layout()
    # #plt.savefig('images/ch8_im7.png')
    # plt.show()
    #
    #
    # # We can also directly inspect the correlation between the features (numerical) and the target:
    #
    # # In[26]:
    #
    #
    # df.select_dtypes(include='number').corr()[['default_payment_next_month']]
    #
    #
    # # 7. Plot the distribution of limit balance for each gender and education level:
    #
    # # In[27]:
    #
    #
    # ax = sns.violinplot(x='education', y='limit_bal',
    #                     hue='sex', split=True, data=df)
    # ax.set_title('Distribution of limit balance per education level',
    #              fontsize=16)
    #
    # plt.tight_layout()
    # # plt.savefig('images/ch8_im8.png')
    # plt.show()
    #
    #
    # # The following code plots the same information, without splitting the violin plots.
    #
    # # In[19]:
    #
    #
    # # ax = sns.violinplot(x='education', y='limit_bal',
    # #                     hue='sex', data=df)
    # # ax.set_title('Distribution of limit balance per education level',
    # #              fontsize=16);
    #
    #
    # # 8. Investigate the distribution of the target variable per gender and education level:
    #
    # # In[29]:
    #
    #
    # ax = sns.countplot('default_payment_next_month', hue='sex',
    #                    data=df, orient='h')
    # ax.set_title('Distribution of the target variable', fontsize=16)
    #
    # plt.tight_layout()
    # # plt.savefig('images/ch8_im9.png')
    # plt.show()
    #
    #
    # # 9. Investigate the percentage of defaults per education level:
    #
    # # In[30]:
    #
    #
    # ax = df.groupby('education')['default_payment_next_month']        .value_counts(normalize=True)        .unstack()        .plot(kind='barh', stacked='True')
    # ax.set_title('Percentage of default per education level',
    #              fontsize=16)
    # ax.legend(title='Default', bbox_to_anchor=(1,1))
    #
    # plt.tight_layout()
    # # plt.savefig('images/ch8_im10.png')
    # plt.show()
    #
    #
    # # ### There's more
    #
    # # In[ ]:
    #
    #
    # # import pandas_profiling
    # # df.profile_report()
    #
    #
    # # ## Splitting the data into training and test sets
    #
    # # ### How to do it...
    #
    # # 1. Import the function from `sklearn`:
    #
    # # In[15]:
    #
    #
    # from sklearn.model_selection import train_test_split
    #
    #
    # # 2. Split the data into training and test sets:
    #
    # # In[22]:
    #
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                     test_size=0.2,
    #                                                     random_state=42)
    #
    #
    # # 3. Split the data into training and test sets without shuffling:
    #
    # # In[23]:
    #
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                     test_size=0.2,
    #                                                     shuffle=False)
    #
    #
    # # 4. Split the data into training and test sets with stratification:
    #
    # # In[16]:
    #
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                     test_size=0.2,
    #                                                     stratify=y,
    #                                                     random_state=42)
    #
    #
    # # 5. Verify that the ratio of the target is preserved:
    #
    # # In[25]:
    #
    #
    # y_train.value_counts(normalize=True)
    #
    #
    # # In[26]:
    #
    #
    # y_test.value_counts(normalize=True)
    #
    #
    # # ### There's more
    #
    # # In[27]:
    #
    #
    # # # define the size of the validation and test sets
    # # VALID_SIZE = 0.1
    # # TEST_SIZE = 0.2
    #
    # # # create the initial split - training and temp
    # # X_train, X_temp, y_train, y_temp = train_test_split(X, y,
    # #                                                     test_size=(VALID_SIZE + TEST_SIZE),
    # #                                                     stratify=y,
    # #                                                     random_state=42)
    #
    # # # calculate the new test size
    # # NEW_TEST_SIZE = np.around(TEST_SIZE / (VALID_SIZE + TEST_SIZE), 2)
    #
    # # # create the valid and test sets
    # # X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp,
    # #                                                     test_size=NEW_TEST_SIZE,
    # #                                                     stratify=y_temp,
    # #                                                     random_state=42)
    #
    #
    # # ## Dealing with missing values
    #
    # # ### How to do it...
    #
    # # 1. Import the libraries:
    #
    # # In[17]:
    #
    #
    # import pandas as pd
    # import missingno
    # from sklearn.impute import SimpleImputer
    #
    #
    # # 2. Inspect the information about the DataFrame:
    #
    # # In[18]:
    #
    #
    # X.info()
    #
    #
    # # 3. Visualize the nullity of the DataFrame:
    #
    # # In[19]:
    #
    #
    # missingno.matrix(X)
    #
    # # plt.savefig('images/ch8_im12.png')
    # plt.show()
    #
    #
    # # 4. Define columns with missing values per data type:
    #
    # # In[20]:
    #
    #
    # NUM_FEATURES = ['age']
    # CAT_FEATURES = ['sex', 'education', 'marriage']
    #
    #
    # # 5. Impute the numerical feature:
    #
    # # In[21]:
    #
    #
    # for col in NUM_FEATURES:
    #     num_imputer = SimpleImputer(strategy='median')
    #     num_imputer.fit(X_train[[col]])
    #     X_train.loc[:, col] = num_imputer.transform(X_train[[col]])
    #     X_test.loc[:, col] = num_imputer.transform(X_test[[col]])
    #
    #
    # # In[22]:
    #
    #
    # # alternative method using pandas
    #
    # # for feature in NUM_FEATURES:
    # #     median_value = X_train[feature].median()
    # #     X_train.loc[:, feature].fillna(median_value, inplace=True)
    # #     X_test.loc[:, feature].fillna(median_value, inplace=True)
    #
    #
    # # 6. Impute the categorical features:
    #
    # # In[23]:
    #
    #
    # for col in CAT_FEATURES:
    #     cat_imputer = SimpleImputer(strategy='most_frequent')
    #     cat_imputer.fit(X_train[[col]])
    #     X_train.loc[:, col] = cat_imputer.transform(X_train[[col]])
    #     X_test.loc[:, col] = cat_imputer.transform(X_test[[col]])
    #
    #
    # # In[24]:
    #
    #
    # # alternative method using pandas
    #
    # # for feature in CAT_FEATURES:
    # #     mode_value = X_train[feature].mode().values[0]
    # #     X_train.loc[:, feature].fillna(mode_value, inplace=True)
    # #     X_test.loc[:, feature].fillna(mode_value, inplace=True)
    #
    #
    # # 7. Verify that there are no missing values:
    #
    # # In[25]:
    #
    #
    # X_train.info()
    #
    #
    # # ## Encoding categorical variables
    #
    # # ### How to do it...
    #
    # # 1. Import the libraries:
    #
    # # In[34]:
    #
    #
    # import pandas as pd
    # from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # from sklearn.compose import ColumnTransformer
    #
    #
    # # 2. Use Label Encoder to encode a selected column:
    #
    # # In[35]:
    #
    #
    # COL = 'education'
    #
    # X_train_copy = X_train.copy()
    # X_test_copy = X_test.copy()
    #
    # label_enc = LabelEncoder()
    # label_enc.fit(X_train_copy[COL])
    # X_train_copy.loc[:, COL] = label_enc.transform(X_train_copy[COL])
    # X_test_copy.loc[:, COL] = label_enc.transform(X_test_copy[COL])
    #
    #
    # # 3. Select categorical features for one-hot encoding:
    #
    # # In[36]:
    #
    #
    # CAT_FEATURES = X_train.select_dtypes(include='object')                       .columns                       .to_list()
    #
    #
    # # 4. Instantiate the One-Hot Encoder object:
    #
    # # In[37]:
    #
    #
    # one_hot_encoder = OneHotEncoder(sparse=False,
    #                                 handle_unknown='error',
    #                                 drop='first')
    #
    #
    # # 5. Create the column transformer using the one-hot encoder:
    #
    # # In[38]:
    #
    #
    # one_hot_transformer = ColumnTransformer(
    #     [("one_hot", one_hot_encoder, CAT_FEATURES)]
    #     #,remainder='passthrough'
    # )
    #
    #
    # # 6. Fit the transformer:
    #
    # # In[39]:
    #
    #
    # one_hot_transformer.fit(X_train)
    #
    #
    # # 7. Apply the transformations to both training and test sets:
    #
    # # In[40]:
    #
    #
    # col_names = one_hot_transformer.get_feature_names()
    #
    # X_train_cat = pd.DataFrame(one_hot_transformer.transform(X_train),
    #                            columns=col_names,
    #                            index=X_train.index)
    # X_train_ohe = pd.concat([X_train, X_train_cat], axis=1)                 .drop(CAT_FEATURES, axis=1)
    #
    # X_test_cat = pd.DataFrame(one_hot_transformer.transform(X_test),
    #                           columns=col_names,
    #                           index=X_test.index)
    # X_test_ohe = pd.concat([X_test, X_test_cat], axis=1)                .drop(CAT_FEATURES, axis=1)
    #
    #
    # # ### There's more
    #
    # # #### Using `pandas.get_dummies` for one-hot encoding
    #
    # # In[53]:
    #
    #
    # pd.get_dummies(X_train, prefix_sep='_', drop_first=True)
    #
    #
    # # #### Specifying possible categories for OneHotEncoder
    #
    # # In[54]:
    #
    #
    # one_hot_encoder = OneHotEncoder(
    #     categories=[['Male', 'Female', 'Unknown']],
    #     sparse=False,
    #     handle_unknown='error',
    #     drop='first'
    # )
    #
    # one_hot_transformer = ColumnTransformer(
    #     [("one_hot", one_hot_encoder, ['sex'])]
    # )
    #
    # one_hot_transformer.fit(X_train)
    # one_hot_transformer.get_feature_names()
    #
    #
    # # #### Category Encoders library
    #
    # # In[55]:
    #
    #
    # import category_encoders as ce
    #
    #
    # # In[56]:
    #
    #
    # one_hot_encoder_ce = ce.OneHotEncoder(use_cat_names=True)
    #
    #
    # # In[57]:
    #
    #
    # one_hot_encoder_ce.fit(X_train)
    # X_train_ce = one_hot_encoder_ce.transform(X_train)
    # X_train_ce.head()
    #
    #
    # # In[58]:
    #
    #
    # target_encoder = ce.TargetEncoder(smoothing=0)
    # target_encoder.fit(X_train.sex, y_train)
    # target_encoder.transform(X_train.sex).head()
    #
    #
    # # ## Fitting a decision tree classifier
    #
    # # ### How to do it...
    #
    # # 1. Import the libraries:
    #
    # # In[41]:
    #
    #
    # from sklearn.tree import DecisionTreeClassifier, export_graphviz
    # from sklearn import metrics
    #
    # from chapter_8_utils import performance_evaluation_report
    #
    # from io import StringIO
    # import seaborn as sns
    # from ipywidgets import Image
    # import pydotplus
    #
    #
    # # 2. Create the instance of the model, fit it to the training data and create prediction:
    #
    # # In[42]:
    #
    #
    # tree_classifier = DecisionTreeClassifier(random_state=42)
    # tree_classifier.fit(X_train_ohe, y_train)
    # y_pred = tree_classifier.predict(X_test_ohe)
    #
    #
    # # 3. Evaluate the results:
    #
    # # In[43]:
    #
    #
    # LABELS = ['No Default', 'Default']
    # tree_perf = performance_evaluation_report(tree_classifier,
    #                                           X_test_ohe,
    #                                           y_test, labels=LABELS,
    #                                           show_plot=True)
    #
    # plt.tight_layout()
    # # plt.savefig('images/ch8_im14.png')
    # plt.show()
    #
    #
    # # In[44]:
    #
    #
    # tree_perf
    #
    #
    # # 4. Plot the simplified Decision Tree:
    #
    # # In[66]:
    #
    #
    # small_tree = DecisionTreeClassifier(max_depth=3,
    #                                     random_state=42)
    # small_tree.fit(X_train_ohe, y_train)
    #
    # tree_dot = StringIO()
    # export_graphviz(small_tree, feature_names=X_train_ohe.columns,
    #                 class_names=LABELS, rounded=True, out_file=tree_dot,
    #                 proportion=False, precision=2, filled=True)
    # tree_graph = pydotplus.graph_from_dot_data(tree_dot.getvalue())
    # tree_graph.set_dpi(300)
    # # tree_graph.write_png('images/ch8_im15.png')
    # Image(value=tree_graph.create_png())
    #
    #
    # # ### There's more
    #
    # # In[45]:
    #
    #
    # y_pred_prob = tree_classifier.predict_proba(X_test_ohe)[:, 1]
    #
    #
    # # In[46]:
    #
    #
    # precision, recall, thresholds = metrics.precision_recall_curve(y_test,
    #                                                                y_pred_prob)
    #
    #
    # # In[47]:
    #
    #
    # ax = plt.subplot()
    # ax.plot(recall, precision,
    #         label=f'PR-AUC = {metrics.auc(recall, precision):.2f}')
    # ax.set(title='Precision-Recall Curve',
    #        xlabel='Recall',
    #        ylabel='Precision')
    # ax.legend()
    #
    # plt.tight_layout()
    # # plt.savefig('images/ch8_im16.png')
    # plt.show()
    #
    #
    # # ## Implementing scikit-learn's pipelines
    #
    # # ### How to do it...
    #
    # # 1. Import the libraries:
    #
    # # In[1]:
    #
    #
    # import pandas as pd
    # from sklearn.model_selection import train_test_split
    # from sklearn.impute import SimpleImputer
    # from sklearn.preprocessing import OneHotEncoder
    # from sklearn.compose import ColumnTransformer
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.pipeline import Pipeline
    # from chapter_8_utils import performance_evaluation_report
    #
    #
    # # 2. Load the data, separate the target and create the stratified train-test split:
    #
    # # In[2]:
    #
    #
    # df = pd.read_csv('../Datasets/credit_card_default.csv',
    #                  index_col=0, na_values='')
    #
    # X = df.copy()
    # y = X.pop('default_payment_next_month')
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                     test_size=0.2,
    #                                                     stratify=y,
    #                                                     random_state=42)
    #
    #
    # # 3. Store lists of numerical/categorical features:
    #
    # # In[3]:
    #
    #
    # num_features = X_train.select_dtypes(include='number')                       .columns                       .to_list()
    # cat_features = X_train.select_dtypes(include='object')                       .columns                       .to_list()
    #
    #
    # # 4. Define the numerical pipeline:
    #
    # # In[4]:
    #
    #
    # num_pipeline = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='median'))
    # ])
    #
    #
    # # 5. Define the categorical pipeline:
    #
    # # In[5]:
    #
    #
    # cat_list = [list(X_train[col].dropna().unique()) for col in cat_features]
    #
    # cat_pipeline = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='most_frequent')),
    #     ('onehot', OneHotEncoder(categories=cat_list, sparse=False,
    #                              handle_unknown='error', drop='first'))
    # ])
    #
    #
    # # 6. Define the column transformer object:
    #
    # # In[6]:
    #
    #
    # preprocessor = ColumnTransformer(transformers=[
    #     ('numerical', num_pipeline, num_features),
    #     ('categorical', cat_pipeline, cat_features)],
    #     remainder='drop')
    #
    #
    # # 7. Create the joint pipeline:
    #
    # # In[7]:
    #
    #
    # dec_tree = DecisionTreeClassifier(random_state=42)
    #
    # tree_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
    #                                 ('classifier', dec_tree)])
    #
    #
    # # 8. Fit the pipeline to the data:
    #
    # # In[8]:
    #
    #
    # tree_pipeline.fit(X_train, y_train)
    #
    #
    # # 9. Evaluate the performance of the entire pipeline:
    #
    # # In[12]:
    #
    #
    # LABELS = ['No Default', 'Default']
    # tree_perf = performance_evaluation_report(tree_pipeline, X_test,
    #                                           y_test, labels=LABELS,
    #                                           show_plot=True)
    #
    # plt.tight_layout()
    # # plt.savefig('images/ch8_im17.png')
    # plt.show()
    #
    #
    # # In[12]:
    #
    #
    # tree_perf
    #
    #
    # # ### There's more
    #
    # # 1. Import the base estimator and transformer from `sklearn`:
    #
    # # In[13]:
    #
    #
    # from sklearn.base import BaseEstimator, TransformerMixin
    #
    #
    # # 2. Define the `OutlierRemover` class:
    #
    # # In[14]:
    #
    #
    # class OutlierRemover(BaseEstimator, TransformerMixin):
    #     def __init__(self, n_std=3):
    #         self.n_std = n_std
    #
    #     def fit(self, X, y = None):
    #         if np.isnan(X).any(axis=None):
    #             raise ValueError('''There are missing values in the array!
    #                                 Please remove them.''')
    #
    #         mean_vec = np.mean(X, axis=0)
    #         std_vec = np.std(X, axis=0)
    #
    #         self.upper_band_ = mean_vec + self.n_std * std_vec
    #         self.lower_band_ = mean_vec - self.n_std * std_vec
    #         self.n_features_ = len(self.upper_band_)
    #
    #         return self
    #
    #     def transform(self, X, y = None):
    #         X_copy = pd.DataFrame(X.copy())
    #
    #         upper_band = np.repeat(
    #             self.upper_band_.reshape(self.n_features_, -1),
    #             len(X_copy),
    #             axis=1).transpose()
    #         lower_band = np.repeat(
    #             self.lower_band_.reshape(self.n_features_, -1),
    #             len(X_copy),
    #             axis=1).transpose()
    #
    #         X_copy[X_copy >= upper_band] = upper_band
    #         X_copy[X_copy <= lower_band] = lower_band
    #
    #         return X_copy.values
    #
    #
    # # 3. Add the `OutlierRemover` to the numerical Pipeline:
    #
    # # In[15]:
    #
    #
    # num_pipeline = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='median')),
    #     ('outliers', OutlierRemover())
    # ])
    #
    #
    # # 4. Run the rest of the Pipeline, to compare the results:
    #
    # # In[16]:
    #
    #
    # preprocessor = ColumnTransformer(transformers=[
    #     ('numerical', num_pipeline, num_features),
    #     ('categorical', cat_pipeline, cat_features)],
    #     remainder='drop')
    #
    # dec_tree = DecisionTreeClassifier(random_state=42)
    #
    # tree_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
    #                                 ('classifier', dec_tree)])
    #
    # tree_pipeline.fit(X_train, y_train)
    #
    # tree_perf = performance_evaluation_report(tree_pipeline, X_test,
    #                                           y_test, labels=LABELS,
    #                                           show_plot=True)
    #
    # plt.tight_layout()
    # # plt.savefig('images/ch8_im18.png')
    # plt.show()
    #
    #
    # # In[17]:
    #
    #
    # tree_perf
    #
    #
    # # ## Tuning hyperparameters using grid search and cross-validation
    #
    # # ### How to do it...
    #
    # # 1. Import the libraries:
    #
    # # In[18]:
    #
    #
    # from sklearn.model_selection import (GridSearchCV, cross_val_score,
    #                                      RandomizedSearchCV, cross_validate,
    #                                      StratifiedKFold)
    # from sklearn import metrics
    #
    #
    # # 2. Define the cross-validation scheme:
    #
    # # In[19]:
    #
    #
    # k_fold = StratifiedKFold(5, shuffle=True, random_state=42)
    #
    #
    # # 3. Evaluate the pipeline using cross-validation:
    #
    # # In[20]:
    #
    #
    # cross_val_score(tree_pipeline, X_train, y_train, cv=k_fold)
    #
    #
    # # 4. Add extra metrics to cross-validation:
    #
    # # In[21]:
    #
    #
    # cross_validate(tree_pipeline, X_train, y_train, cv=k_fold,
    #                scoring=['accuracy', 'precision', 'recall',
    #                         'roc_auc'])
    #
    #
    # # 5. Define the parameter grid:
    #
    # # In[22]:
    #
    #
    # param_grid = {'classifier__criterion': ['entropy', 'gini'],
    #               'classifier__max_depth': range(3, 11),
    #               'classifier__min_samples_leaf': range(2, 11),
    #               'preprocessor__numerical__outliers__n_std': [3, 4]}
    #
    #
    # # 6. Run Grid Search:
    #
    # # In[23]:
    #
    #
    # classifier_gs = GridSearchCV(tree_pipeline, param_grid, scoring='recall',
    #                              cv=k_fold, n_jobs=-1, verbose=1)
    #
    # classifier_gs.fit(X_train, y_train)
    #
    #
    # # In[24]:
    #
    #
    # print(f'Best parameters: {classifier_gs.best_params_}')
    # print(f'Recall (Training set): {classifier_gs.best_score_:.4f}')
    # print(f'Recall (Test set): {metrics.recall_score(y_test, classifier_gs.predict(X_test)):.4f}')
    #
    #
    # # 7. Evaluate the performance of the Grid Search:
    #
    # # In[25]:
    #
    #
    # LABELS = ['No Default', 'Default']
    # tree_gs_perf = performance_evaluation_report(classifier_gs, X_test,
    #                                              y_test, labels=LABELS,
    #                                              show_plot=True)
    #
    # plt.tight_layout()
    # plt.savefig('images/ch8_im20.png')
    # plt.show()
    #
    #
    # # In[26]:
    #
    #
    # tree_gs_perf
    #
    #
    # # 8. Run Randomized Grid Search:
    #
    # # In[27]:
    #
    #
    # classifier_rs = RandomizedSearchCV(tree_pipeline, param_grid, scoring='recall',
    #                                    cv=k_fold, n_jobs=-1, verbose=1,
    #                                    n_iter=100, random_state=42)
    # classifier_rs.fit(X_train, y_train)
    #
    #
    # # In[28]:
    #
    #
    # print(f'Best parameters: {classifier_rs.best_params_}')
    # print(f'Recall (Training set): {classifier_rs.best_score_:.4f}')
    # print(f'Recall (Test set): {metrics.recall_score(y_test, classifier_rs.predict(X_test)):.4f}')
    #
    #
    # # 9. Evaluate the performance of the Randomized Grid Search:
    #
    # # In[29]:
    #
    #
    # tree_rs_perf = performance_evaluation_report(classifier_rs, X_test,
    #                                              y_test, labels=LABELS,
    #                                              show_plot=True)
    #
    # plt.tight_layout()
    # plt.savefig('images/ch8_im21.png')
    # plt.show()
    #
    #
    # # In[30]:
    #
    #
    # tree_rs_perf
    #
    #
    # # ### There's more
    #
    # # In[31]:
    #
    #
    # from sklearn.linear_model import LogisticRegression
    #
    #
    # # In[32]:
    #
    #
    # param_grid = [{'classifier': [LogisticRegression()],
    #                'classifier__penalty': ['l1', 'l2'],
    #                'classifier__C': np.logspace(0, 3, 10, 2),
    #                'preprocessor__numerical__outliers__n_std': [3, 4]},
    #               {'classifier': [DecisionTreeClassifier(random_state=42)],
    #                'classifier__criterion': ['entropy', 'gini'],
    #                'classifier__max_depth': range(3, 11),
    #                'classifier__min_samples_leaf': range(2, 11),
    #                'preprocessor__numerical__outliers__n_std': [3, 4]}]
    #
    #
    # # In[33]:
    #
    #
    # classifier_gs_2 = GridSearchCV(tree_pipeline, param_grid, scoring='recall',
    #                                cv=k_fold, n_jobs=-1, verbose=1)
    #
    # classifier_gs_2.fit(X_train, y_train)
    #
    # print(f'Best parameters: {classifier_gs_2.best_params_}')
    # print(f'Recall (Training set): {classifier_gs_2.best_score_:.4f}')
    # print(f'Recall (Test set): {metrics.recall_score(y_test, classifier_gs_2.predict(X_test)):.4f}')
    #
    #
    # # In[ ]:
    #
    #
    #
    #
    #
    # # In[ ]:
    #
    #
    #
