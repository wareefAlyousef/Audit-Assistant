import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
# from IPython.display import display 
# import ipywidgets as widgets

df = pd.DataFrame()

# # Function to read uploaded file
# def get_uploaded_file(uploader):
#     if len(uploader.value) > 0:
#         file_info = list(uploader.value.values())[0]
#         content = file_info['content']
#         filename = file_info['metadata']['name']

#         # Handle different file types
#         if filename.endswith('.csv'):
#             df = pd.read_csv(pd.io.common.BytesIO(content))
#         elif filename.endswith('.xlsx'):
#             df = pd.read_excel(pd.io.common.BytesIO(content))
#         elif filename.endswith('.parquet'):
#             df = pd.read_parquet(pd.io.common.BytesIO(content))
#         else:
#             raise ValueError("Unsupported file type.")
#         return df
#     else:
#         print("Please upload a file first.")
#         return None


def read_excel_csv(file_path):
    """
    Reads a CSV file exported from Excel and returns a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
def read_excel_xlsx(file_path):
    """
    Reads an Excel file (.xlsx) and returns a pandas DataFrame.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame containing the Excel data.
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None


def normalize_iqr(df):
    """
    Normalize numeric columns of a DataFrame using median and IQR scaling.
    
    Formula:
        if IQR == 0:
            normalized = value - median
        else:
            normalized = (value - median) / IQR
    """
    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        median = df[col].median()
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)

        if iqr == 0:
            df_normalized[col] = df[col] - median
        else:
            df_normalized[col] = (df[col] - median) / iqr

    return df_normalized 


def encode_categoricals(df):
    """
    Encode categorical (non-numeric) columns in the DataFrame using Label Encoding.
    
    Each categorical column is transformed into integer codes.
    """
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(exclude="number").columns

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded


def predict_fraud(df, model_path="catboost.joblib"):
    """
    Load a CatBoost model and predict fraud on the given DataFrame.
    Returns both predictions and a fraud-only DataFrame.
    """
    # Load trained model
    model = joblib.load(model_path)
    
    # Make predictions
    preds = model.predict(df)
    
    # Add predictions as a new column
    df_with_preds = df.copy()
    df_with_preds["predicted_fraud"] = preds
    
    # Fraud-only DataFrame (prediction == 1)
    fraud_df = df_with_preds[df_with_preds["predicted_fraud"] == 1]
    
    return df_with_preds, fraud_df

if __name__ == "__main__":

    # File uploader widget
    # df = get_uploaded_file(widgets.FileUpload(accept='.csv,.xlsx,.parquet', multiple=False))
    file_path = "data.csv"  
    df = read_excel_csv(file_path)

    if df is not None:
        df_normalized = normalize_iqr(df)
        df_encoded = encode_categoricals(df_normalized)
        df_with_preds, fraud_df = predict_fraud(df_encoded)
        
        print("All Predictions:")
        print(df_with_preds)
        
        print("\nFraud Predictions Only:")
        print(fraud_df)