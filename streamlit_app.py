import streamlit as st
import pandas as pd
# First, we need these modules for the basics of data manipulation
import numpy as np


# This part is for the plotting
import matplotlib.pyplot as plt
import seaborn as sns

# This one is just to access the dataset
#from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from catboost import CatBoostClassifier
import shap


st.header("Research question")
st.write("How does cigarette smoking (CURSMOKE) and cholesterol (TOTCHOL, HDLC, LDLC) impact the incidence of coronary heart disease (ANYCHD)?")

st.header("Dataframe description", divider= "blue")
df = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

st.write(df.describe())

columns_to_check = ["CURSMOKE", "ANYCHD", "AGE", "SEX", "BMI", "TOTCHOL", "HDLC", "LDLC"]


columns_to_check1 = ["CURSMOKE", "ANYCHD", "AGE", "SEX", "BMI", "TOTCHOL"]

mean_values = df[columns_to_check].mean()
df_outliers = df.copy()
# Fill NaN values in the specified columns with the calculated means
df.loc[:, columns_to_check] = df.loc[:, columns_to_check].fillna(mean_values)

st.header("Zero values after replacing with the mean", divider= "blue")
col_imp1, col_imp2 = st.columns(2)
columns_to_check2 = ["AGE", "SEX", "BMI", "TOTCHOL"]
(df[columns_to_check2] == 0).sum()
with col_imp1:
    st.write(df[columns_to_check1].isnull().sum())
with col_imp2:
    st.write(df_outliers[columns_to_check1].isnull().sum())
#remove outliers in cholesterol and BMI of whole dataset.
df.loc[(df['TOTCHOL'] <= 100) | (df['TOTCHOL'] > 400), 'TOTCHOL'] = np.nan
df.loc[(df['BMI'] <= 10) | (df['BMI'] >= 50), 'BMI'] = np.nan


#replace the missing values with the mean value in BMI and cholesterol
df.TOTCHOL = df.TOTCHOL.fillna(df.TOTCHOL.mean())
df.BMI = df.BMI.fillna(df.BMI.mean())


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2).set_output(transform = "pandas")
df2 = imputer.fit_transform(df)
df.CIGPDAY = df2.CIGPDAY

Before_followup = df[df.PERIOD == 1]
First_followup = df[df.PERIOD == 2]
Second_followup = df[df.PERIOD == 3]


#Remove the outliers from HDLC and LDLC in the second follow up as this is the only measuring point.
Second_followup.loc[(Second_followup['HDLC'] <= 35) | (Second_followup['HDLC'] > 80), 'HDLC'] = np.nan
Second_followup.loc[(Second_followup['LDLC'] <= 40) | (Second_followup['LDLC'] > 200), 'LDLC'] = np.nan

#Replace the missing values with the mean values.
Second_followup.HDLC.fillna(Second_followup.HDLC.mean())
Second_followup.LDLC.fillna(Second_followup.LDLC.mean())

columns_to_check3 = ["HDLC", "LDLC"]
mean_values = Second_followup[columns_to_check3].mean()

# Fill NaN values in the specified columns with the calculated means
Second_followup.loc[:, columns_to_check3] = Second_followup.loc[:, columns_to_check3].fillna(mean_values)
df.loc[df.PERIOD == 3]["LDLC"] = Second_followup["LDLC"]
df.loc[df.PERIOD == 3]["HDLC"] = Second_followup["HDLC"]
st.header("Zero values for HDLC and LC", divider= "blue")
col_h_ldlc1, col_h_ldlc2 = st.columns(2)
with col_h_ldlc1:
    st.write((Second_followup[columns_to_check3] == 0).sum())
with col_h_ldlc2:
    st.write((df_outliers.loc[df_outliers.PERIOD == 3][columns_to_check3] == 0).sum())

st.header("Histograms before and after outlier removal", divider = 'blue')
st.write("Outliers were removed for the features total cholesterol, BMI, HDLC, and LDLC")
st.subheader("Outlier for removal total cholesterol")
st.write("Total cholesterol values above 400 and below 100 were removed from the dataset, and replaced with the mean")

col1, col2 = st.columns(2)

with col1:
    # Generate some data
    plt.figure(figsize=(8, 6))
    sns.histplot(df_outliers["TOTCHOL"], kde=True, color='blue', bins=20)
    plt.title('Histogram of total cholesterol')
    plt.xlabel("TOTCHOL")
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())

with col2:
    plt.figure(figsize=(8, 6))
    sns.histplot(df["TOTCHOL"], kde=True, color='blue', bins=20)
    plt.title('Histogram of total cholesterol')
    plt.xlabel("TOTCHOL")
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())

st.subheader("Outlier removal for total BMI")
st.write("BMI values above 50 and below 10 were removed from the dataset, and replaced with the mean")

col3, col4 = st.columns(2)

with col3:
    # Generate some data
    plt.figure(figsize=(8, 6))
    sns.histplot(df_outliers["BMI"], kde=True, color='blue', bins=20)
    plt.title('Histogram of BMI')
    plt.xlabel("BMI")
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())

with col4:
    plt.figure(figsize=(8, 6))
    sns.histplot(df["BMI"], kde=True, color='blue', bins=20)
    plt.title('Histogram of BMI')
    plt.xlabel("BMI")
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())


st.subheader("Outlier removal for total HDLC")
st.write("HDLC values above 80 and below 35 were removed from the dataset, and replaced with the mean")
col5, col6 = st.columns(2)

with col5:
    # Generate some data
    plt.figure(figsize=(8, 6))
    sns.histplot(df_outliers.loc[df.PERIOD == 3]["HDLC"], kde=True, color='blue', bins=20)
    plt.title('Histogram of HDLC')
    plt.xlabel("HDLC")
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())

with col6:
    plt.figure(figsize=(8, 6))
    sns.histplot(df.loc[df.PERIOD == 3]["HDLC"], kde=True, color='blue', bins=20)
    plt.title('Histogram of HDLC')
    plt.xlabel("HDLC")
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())

st.subheader("Outlier removal for total LDLC")
st.write("LDLC values above 200 and below 40 were removed from the dataset, and replaced with the mean")
col7, col8 = st.columns(2)

with col7:
    # Generate some data
    plt.figure(figsize=(8, 6))
    sns.histplot(df_outliers.loc[df.PERIOD == 3]["LDLC"], kde=True, color='blue', bins=20)
    plt.title('Histogram of LDLC')
    plt.xlabel("LDLC")
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())

with col8:
    plt.figure(figsize=(8, 6))
    sns.histplot(df.loc[df.PERIOD == 3]["LDLC"], kde=True, color='blue', bins=20)
    plt.title('Histogram of LDLC')
    plt.xlabel("LDLC")
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())

st.header("Variable and period selection", divider= "blue")

# Add a selectbox for periods
period_options = df['PERIOD'].unique()  # Get the unique periods
selected_period = st.selectbox(
    "Select Period",
    period_options,
    key="period"
)

# Filter the dataframe based on the selected period
filtered_df = df[df['PERIOD'] == selected_period]
if selected_period == 3:
    columns_to_check = ["CURSMOKE", "CIGPDAY", "ANYCHD", "AGE", "SEX", "BMI", "TOTCHOL", "HDLC", "LDLC"]
else:
    columns_to_check = ["CURSMOKE", "CIGPDAY", "ANYCHD", "AGE", "SEX", "BMI", "TOTCHOL"]
# Assuming `df` is your dataframe and `columns_to_check` is a list of columns
y_var1 = st.selectbox(
    "Choose Y variable",
    columns_to_check,
    index=None,
    placeholder="Select variable Y",
    key="original1"
)

x_var1 = st.selectbox(
    "Choose X variable",
    columns_to_check,
    index=None,
    placeholder="Select variable X",
    key="original2"
)

bp_col, cp_col, his_col = st.columns(3)
with bp_col:
    cb_bp = st.checkbox("Make boxplot", key="checkboxbp")
with cp_col:
    cb_cp = st.checkbox("Make correlation plot", key="checkboxcp")
with his_col:
    cb_his = st.checkbox("Make histogram", key="checkboxhis")

if cb_bp:
    st.header(f"Boxplot for {y_var1} vs {x_var1} for period {selected_period}", divider= "blue")
    # Create the boxplot for the selected period
    plt.figure(figsize=(8, 6))
    #test1 = sns.boxplot(data=filtered_df, x=x_var1, y=y_var1)
    sns.boxplot(data=filtered_df, x=x_var1, y=y_var1)
    #st.pyplot(test1.get_figure())
    st.pyplot(plt.gcf())

if cb_cp:
    st.header(f"Correlation plot  for {y_var1} vs {x_var1} for period {selected_period}", divider= "blue")
    if x_var1 != None and y_var1 != None:
        # If you want to plot a regression line for the selected period
        for i, group in filtered_df.groupby('PERIOD'):
            sns.lmplot(x=x_var1, y=y_var1, data=group, fit_reg=True)
            st.pyplot(plt.gcf())  # Display the plot for each period group
    else:
        st.write("Please select both an X and a Y variable")
if cb_his:
    st.header(f"Histogram  of {x_var1} for period {selected_period}", divider= "blue")
    if x_var1 != None:
        
        plt.figure(figsize=(8, 6))
        sns.histplot(filtered_df[x_var1], kde=True, color='blue', bins=10)
        plt.title(f'Histogram of {x_var1} for Period: {selected_period}')
        plt.xlabel(x_var1)
        plt.ylabel('Frequency')
        st.pyplot(plt.gcf())
    else:
        st.write("Please select an X variable")


st.header("Heatmap for all variables")
numeric_df = df[columns_to_check].select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(25, 25))
sns.heatmap(numeric_df.corr(), annot=True)
st.pyplot(plt.gcf())



tot_df_model = df[["CURSMOKE", "ANYCHD", "AGE", "SEX", "BMI", "TOTCHOL", "HDLC", "LDLC", "PERIOD"]]

# Define a model function
def model(classifier, df_to_model):
    X = df_to_model.drop(columns=['ANYCHD', "PERIOD"])
    y = df_to_model['ANYCHD']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    return X_train, X_test, y_train, y_test, prediction

# Accuracy metrics function
def accuracy_metrics(y_test, prediction, classifier, X_train, y_train, X_test):
    # Accuracy
    accuracy = accuracy_score(y_test, prediction)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    report = classification_report(y_test, prediction, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("Classification Report:")
    st.dataframe(report_df)

    # Cross Validation Score
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv_score = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    st.write(f"Cross Validation Score (ROC AUC): {cv_score * 100:.2f}%")

    # ROC Curve
    st.write("ROC Curve:")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(classifier, X_test, y_test, ax=ax)
    st.pyplot(fig)

# Confusion Matrix function
def Confusion_matrix(model, X_test, y_test, ax=None):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

#SHAP values function
def Shap_stats(classifier, X_test, Xtrain):
    shap_sample=shap.utils.sample(Xtrain,nsamples = 10, random_state=0)
    explainer = shap.KernelExplainer(classifier.predict_proba, shap_sample)
    shap_values = explainer.shap_values(X_test)
    shap_values_for_class = shap_values[:,:,1]
    feature_names = Xtrain.columns
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.Dataframe(X_test, columns = feature_names)
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True,figsize=(6,4), dpi=150)
    shap.summary_plot(shap_values_for_class, X_test, plot_type="dot")
    st.pyplot(fig)


# Streamlit app
st.header("Model Comparison")

# Model selection widget
model_choice = st.selectbox("Choose a prediction model", ["Logistic Regression", "Random Forest", "Neural Network", "Support Vector Machine", "K-nearest neighbour", "Catboost"])

# Define classifiers based on selection
if model_choice == "Logistic Regression":
    classifier = LogisticRegression(random_state=0, C=10, penalty='l2', class_weight='balanced')
elif model_choice == "Random Forest":
    classifier = RandomForestClassifier(random_state=0, n_estimators=100, class_weight='balanced')
elif model_choice == "Neural Network":
    classifier = MLPClassifier(random_state=0, max_iter=500)
elif model_choice == "Support Vector Machine":
    classifier = SVC(random_state=0, kernel='rbf', C=1.0, probability=True)
elif model_choice == "K-nearest neighbour":
    classifier = KNeighborsClassifier(n_neighbors=5)
elif model_choice == "Catboost":
    classifier = CatBoostClassifier(random_state=0, iterations=100, learning_rate=0.1, verbose=False)

period_options = df['PERIOD'].unique()  # Get the unique periods
#period_options = period_options + "Complete_model"
model_period = st.selectbox(
    "Select Period to model",
    period_options,
    key="model_period"
)

df_model_filtered = tot_df_model[tot_df_model['PERIOD'] == model_period]

# Run the selected model
X_train, X_test, y_train, y_test, prediction = model(classifier, df_model_filtered)

# Display accuracy metrics
st.subheader(f"Results for {model_choice}")
accuracy_metrics(y_test, prediction, classifier, X_train, y_train, X_test)

# Display confusion matrix
st.write("Confusion Matrix:")
fig, ax = plt.subplots()
Confusion_matrix(classifier, X_test, y_test, ax=ax)
st.pyplot(fig)

#display SHAP values
st.write("SHAP values:")
Shap_stats(classifier, X_test, X_train)
