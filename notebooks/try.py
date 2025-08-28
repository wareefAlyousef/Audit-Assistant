import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(
    'Bank_Transaction_Fraud_Detection.csv',
    parse_dates = ['Transaction_Date'],    # Converting 'Transaction_Date' to datetime
    dayfirst = True     # Ensuring the date is parsed as dd-mm-yyyy
)
df['Transaction_Time'] = pd.to_timedelta(df['Transaction_Time'])

# Checking number of rows and columns in the dataset
print(df.shape)

# Viweing the datatype of each column along with number of null values in the dataset
print(df.info())
