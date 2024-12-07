import streamlit as st
import pandas as pd
# First, we need these modules for the basics of data manipulation
import numpy as np


# This part is for the plotting
import matplotlib.pyplot as plt
import seaborn as sns

# This one is just to access the dataset
#from sklearn import datasets

# You can set a general style for your plots like this
#plt.style.use('ggplot')
# plt.style.use('seaborn-whitegrid')

#import altair as alt
#alt.data_transformers.disable_max_rows()
#from sklearn.model_selection import train_test_split, cross_validate
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from sklearn.compose import make_column_transformer
#from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import RepeatedStratifiedKFold
#from sklearn.model_selection import cross_val_score
#from sklearn.inspection import permutation_importance
#from sklearn.pipeline import Pipeline
#from sklearn.metrics import RocCurveDisplay
#pd.set_option('display.max_columns', None)
#import warnings
#warnings.filterwarnings("ignore")

st.header("Dataframe description", divider= "blue")
df = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

st.write(df.describe())

columns_to_check = ["CURSMOKE", "CIGPDAY", "STROKE", "ANYCHD", "AGE", "SEX", "BMI", "TOTCHOL", "HDLC", "LDLC"]


columns_to_check1 = ["CURSMOKE", "CIGPDAY", "STROKE", "ANYCHD", "AGE", "SEX", "BMI", "TOTCHOL"]

mean_values = df[columns_to_check].mean()

# Fill NaN values in the specified columns with the calculated means
df.loc[:, columns_to_check] = df.loc[:, columns_to_check].fillna(mean_values)

st.header("Zero values after taking the mean", divider= "blue")
columns_to_check2 = ["AGE", "SEX", "BMI", "TOTCHOL"]
(df[columns_to_check2] == 0).sum()
st.write(df[columns_to_check1].isnull().sum())

#remove outliers in cholesterol and BMI of whole dataset.
df.loc[(df['TOTCHOL'] <= 100) | (df['TOTCHOL'] > 400), 'TOTCHOL'] = np.nan
df.loc[(df['BMI'] <= 10) | (df['BMI'] >= 50), 'BMI'] = np.nan


#replace the missing values with the mean value in BMI and cholesterol
df.TOTCHOL = df.TOTCHOL.fillna(df.TOTCHOL.mean())
df.BMI = df.BMI.fillna(df.BMI.mean())

st.header("Zero values after taking out the outliers and the missing values", divider= "blue")
st.write(df[columns_to_check2].isnull().sum())

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2).set_output(transform = "pandas")
df2 = imputer.fit_transform(df)
df.CIGPDAY = df2.CIGPDAY

Before_followup = df[df.PERIOD == 1]
First_followup = df[df.PERIOD == 2]
Second_followup = df[df.PERIOD == 3]

# Optionally, you can check the size of each DataFrame
print("Participants before followup:", Before_followup.shape[0])
print("Participants in first follow up:", First_followup.shape[0])
print("Participants in second follow up:", Second_followup.shape[0])

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

st.header("Zero values for HDLC and LC", divider= "blue")
st.write((Second_followup[columns_to_check3] == 0).sum())


st.header("Widget", divider= "blue")
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

# Add a selectbox for periods
period_options = df['PERIOD'].unique()  # Get the unique periods
selected_period = st.selectbox(
    "Select Period",
    period_options,
    key="period"
)

# Filter the dataframe based on the selected period
filtered_df = df[df['PERIOD'] == selected_period]

st.header("Boxplot", divider= "blue")
# Create the boxplot for the selected period
test1 = sns.boxplot(data=filtered_df, x=x_var1, y=y_var1, hue="PERIOD")
st.pyplot(test1.get_figure())

st.header("Correlation plot", divider= "blue")
# If you want to plot a regression line for the selected period
for i, group in filtered_df.groupby('PERIOD'):
    sns.lmplot(x=x_var1, y=y_var1, data=group, fit_reg=True)
    st.pyplot(plt.gcf())  # Display the plot for each period group

st.header("Histogram", divider= "blue")
plt.figure(figsize=(8, 6))
sns.histplot(filtered_df[x_var1], kde=True, color='blue', bins=10)
plt.title(f'Histogram of {x_var1} for Period: {selected_period}')
plt.xlabel(x_var1)
plt.ylabel('Frequency')
st.pyplot(plt.gcf())
