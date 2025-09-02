from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
from flask_cors import CORS
import traceback
import numpy as np
from io import BytesIO
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = "catboost_model.joblib"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
REQUIRED_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig"]

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model file '{MODEL_PATH}' not found. Please ensure it's in the same directory as app.py")

# ---------- Helper Functions ----------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_dataframe(df):
    """Check if dataframe has required columns"""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("Uploaded file is empty")
    
    # Check for non-numeric values in numeric columns
    numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Check if there are too many NaN values
            if df[col].isna().sum() > len(df) / 2:  # More than half are NaN
                raise ValueError(f"Column '{col}' contains too many non-numeric values")
    
    return True

def normalize_iqr(df):
    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        # Handle NaN values by filling with median
        if df[col].isna().any():
            median_val = df[col].median()
            df_normalized[col] = df[col].fillna(median_val)
            
        median = df_normalized[col].median()
        iqr = df_normalized[col].quantile(0.75) - df_normalized[col].quantile(0.25)
        
        # Avoid division by zero
        if iqr != 0:
            df_normalized[col] = (df_normalized[col] - median) / iqr
        else:
            df_normalized[col] = df_normalized[col] - median
            
    return df_normalized

def encode_categoricals(df):
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(exclude="number").columns
    
    for col in cat_cols:
        le = LabelEncoder()
        # Handle NaN values by filling with a placeholder
        if df_encoded[col].isna().any():
            df_encoded[col] = df_encoded[col].fillna("UNKNOWN")
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def predict_fraud(df, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    model = joblib.load(model_path)
    # Get probability scores along with predictions
    preds = model.predict(df)
    probas = model.predict_proba(df)
    
    return preds, probas

def cleanup_file(file_path):
    """Remove temporary file after processing"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not remove temporary file {file_path}: {str(e)}")

def generate_analytics(df_result):
    """Generate analytics data from results"""
    analytics = {}
    
    # Basic stats
    analytics['total_count'] = len(df_result)
    analytics['fraud_count'] = int((df_result["predicted_fraud"] == 1).sum())
    analytics['legit_count'] = analytics['total_count'] - analytics['fraud_count']
    analytics['fraud_rate'] = round((analytics['fraud_count'] / analytics['total_count'] * 100), 2) if analytics['total_count'] > 0 else 0
    
    # Fraud by type
    if 'type' in df_result.columns:
        fraud_by_type = df_result[df_result['predicted_fraud'] == 1].groupby('type').size().to_dict()
        analytics['fraud_by_type'] = fraud_by_type
        
        # Most common fraud type
        if fraud_by_type:
            analytics['most_common_fraud_type'] = max(fraud_by_type, key=fraud_by_type.get)
        else:
            analytics['most_common_fraud_type'] = "None"
    
    # Fraud over time (by step)
    if 'step' in df_result.columns:
        fraud_over_time = df_result[df_result['predicted_fraud'] == 1].groupby('step').size().to_dict()
        analytics['fraud_over_time'] = fraud_over_time
    
    # Amount statistics
    if 'amount' in df_result.columns:
        fraud_amounts = df_result[df_result['predicted_fraud'] == 1]['amount']
        if not fraud_amounts.empty:
            analytics['max_fraud_amount'] = round(fraud_amounts.max(), 2)
            analytics['avg_fraud_amount'] = round(fraud_amounts.mean(), 2)
            analytics['min_fraud_amount'] = round(fraud_amounts.min(), 2)
        else:
            analytics['max_fraud_amount'] = 0
            analytics['avg_fraud_amount'] = 0
            analytics['min_fraud_amount'] = 0
    
    return analytics

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file_path = None
    try:
        # Check if file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        
        # Check if filename is empty
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({"error": f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        # Save to temporary uploads folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Read file based on extension
        if file.filename.endswith(".csv"):
            df_original = pd.read_csv(file_path)
        elif file.filename.endswith(".xlsx"):
            df_original = pd.read_excel(file_path)
        
        # Validate the dataframe
        validate_dataframe(df_original)
        
        # Process data
        df_normalized = normalize_iqr(df_original)
        df_encoded = encode_categoricals(df_normalized)
        
        # Select only the required columns for prediction
        df_for_prediction = df_encoded[REQUIRED_COLUMNS].copy()
        
        # Make predictions
        preds, probas = predict_fraud(df_for_prediction)

        # Prepare results
        df_result = df_original.copy()
        df_result["predicted_fraud"] = preds
        df_result["fraud_probability"] = [round(prob[1] * 100, 2) for prob in probas]  # Probability of fraud class

        # Generate analytics
        analytics = generate_analytics(df_result)

        # Prepare JSON for frontend
        expected_cols = REQUIRED_COLUMNS + ["predicted_fraud", "fraud_probability"]
        available_cols = [c for c in expected_cols if c in df_result.columns]

        # Convert to list of dictionaries for JSON response
        data = df_result[available_cols].head(100).replace({np.nan: None}).to_dict(orient="records")

        return jsonify({
            "success": True,
            "data": data,
            "analytics": analytics,
            "message": f"Processed {analytics['total_count']} transactions. Found {analytics['fraud_count']} potential fraud cases ({analytics['fraud_rate']}%)."
        })
        
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except pd.errors.EmptyDataError:
        return jsonify({"error": "The uploaded file is empty"}), 400
    except pd.errors.ParserError:
        return jsonify({"error": "Error parsing the file. Please check the file format."}), 400
    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Error processing file: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    finally:
        # Clean up temporary file
        if file_path:
            cleanup_file(file_path)

@app.route("/download", methods=["POST"])
def download():
    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return jsonify({"error": "No data provided for download"}), 400
        
        # Create DataFrame from provided data
        df = pd.DataFrame(data['results'])
        
        # Create in-memory file
        output = BytesIO()
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fraud_detection_results_{timestamp}.xlsx"
        
        # Save to Excel
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Total Transactions', 'Fraudulent Transactions', 'Legitimate Transactions', 'Fraud Rate'],
                'Value': [
                    data.get('total_count', 0),
                    data.get('fraud_count', 0),
                    data.get('total_count', 0) - data.get('fraud_count', 0),
                    f"{data.get('fraud_rate', 0)}%"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        
        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        app.logger.error(f"Error generating download: {str(e)}")
        return jsonify({"error": f"Error generating download: {str(e)}"}), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "healthy",
        "model_loaded": model_exists,
        "model_path": MODEL_PATH
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_transaction():
    """API endpoint for analyzing a single transaction"""
    try:
        transaction_data = request.get_json()
        if not transaction_data:
            return jsonify({"error": "No transaction data provided"}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Validate
        validate_dataframe(df)
        
        # Process data
        df_normalized = normalize_iqr(df)
        df_encoded = encode_categoricals(df_normalized)
        
        # Select only the required columns for prediction
        df_for_prediction = df_encoded[REQUIRED_COLUMNS].copy()
        
        # Make prediction
        pred, proba = predict_fraud(df_for_prediction)
        
        return jsonify({
            "success": True,
            "prediction": int(pred[0]),
            "fraud_probability": round(proba[0][1] * 100, 2),
            "is_fraud": bool(pred[0] == 1)
        })
        
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Error analyzing transaction: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    # Check if model exists before starting
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file '{MODEL_PATH}' not found. The application will start but prediction functionality will fail.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)