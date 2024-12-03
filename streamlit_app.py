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

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

df = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

df

columns_to_check = ["CURSMOKE", "CIGPDAY", "STROKE", "ANYCHD", "AGE", "SEX", "BMI", "TOTCHOL", "HDLC", "LDLC"]
df[columns_to_check]

columns_to_check1 = ["CURSMOKE", "CIGPDAY", "STROKE", "ANYCHD", "AGE", "SEX", "BMI", "TOTCHOL"]
df[columns_to_check1].isnull().sum()

mean_values = df[columns_to_check].mean()

# Fill NaN values in the specified columns with the calculated means
df.loc[:, columns_to_check] = df.loc[:, columns_to_check].fillna(mean_values)

columns_to_check2 = ["AGE", "SEX", "BMI", "TOTCHOL"]
(df[columns_to_check2] == 0).sum()

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


y_var = st.selectbox(
    "How would you like to be contacted?",
    (columns_to_check),
    index=None,
    placeholder="Select contact method...",
)
test = sns.boxplot(data = df, x = "ANYCHD", y = y_var, hue="PERIOD")
st.pyplot(test.get_figure())

for i, group in df.groupby('PERIOD'):
    sns.lmplot(x="CIGPDAY", y="STROKE", data=group, fit_reg=True)
plt.show









