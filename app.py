"""
Flask Backend Application for Mobile Money Fraud Detection System.
Provides API endpoints for fraud analysis using hybrid unsupervised ML models.
"""

import os
import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import backend modules
from backend.preprocessing import preprocess_dataset, get_dataset_summary, load_and_clean_dataset
from backend.models import train_all_models
from backend.evaluation import evaluate_all_models, get_anomaly_scores_over_time
from backend.utils import (
    validate_file, 
    validate_dataset_columns, 
    format_flagged_transactions,
    create_error_response,
    create_success_response
)

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Configuration
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


@app.route('/')
def serve_frontend():
    """Serve the frontend HTML page."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from frontend directory."""
    return send_from_directory(app.static_folder, path)


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response indicating backend status
    """
    return jsonify({
        "status": "healthy",
        "message": "Backend running",
        "version": "1.0.0"
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze uploaded CSV file for fraud detection.
    
    Accepts:
        CSV file upload (PaySim format)
        
    Returns:
        JSON response with analysis results including:
        - dataset_summary
        - performance metrics
        - confusion matrices
        - flagged transactions
        - anomaly scores over time
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify(create_error_response("No file uploaded")[0]), 400
        
        file = request.files['file']
        
        # Validate file
        is_valid, error_msg = validate_file(file)
        if not is_valid:
            return jsonify(create_error_response(error_msg)[0]), 400
        
        # Read file into memory
        file_content = io.BytesIO(file.read())
        
        # Load and validate dataset structure
        try:
            df_raw = load_and_clean_dataset(file_content)
        except Exception as e:
            return jsonify(create_error_response(f"Error reading CSV file: {str(e)}")[0]), 400
        
        # Validate required columns
        is_valid, error_msg = validate_dataset_columns(df_raw)
        if not is_valid:
            return jsonify(create_error_response(error_msg)[0]), 400
        
        # Check if dataset has ground truth labels
        has_labels = 'isFraud' in df_raw.columns
        original_labels = df_raw['isFraud'].values if has_labels else None
        
        # Reset file pointer and preprocess
        file_content.seek(0)
        
        # Preprocess dataset
        try:
            X, df_processed, feature_columns = preprocess_dataset(file_content)
        except Exception as e:
            return jsonify(create_error_response(f"Error preprocessing data: {str(e)}")[0]), 400
        
        # Determine contamination rate
        if has_labels:
            contamination = max(0.01, min(0.5, np.mean(original_labels)))
        else:
            contamination = 0.1  # Default contamination rate
        
        # Train all models
        try:
            model_results = train_all_models(X, contamination=contamination)
        except Exception as e:
            return jsonify(create_error_response(f"Error training models: {str(e)}")[0]), 400
        
        # Evaluate models
        try:
            evaluation_results = evaluate_all_models(model_results, original_labels)
        except Exception as e:
            return jsonify(create_error_response(f"Error evaluating models: {str(e)}")[0]), 400
        
        # Get anomaly scores over time
        step_values = df_processed['step'].values
        anomaly_time_series = get_anomaly_scores_over_time(model_results, step_values)
        
        # Generate dataset summary
        final_predictions = evaluation_results['ensemble_results']['predictions']
        dataset_summary = get_dataset_summary(df_processed, final_predictions)
        
        # Format flagged transactions
        flagged_transactions = format_flagged_transactions(
            df_processed,
            model_results['isolation_forest']['scores'].tolist(),
            model_results['lof']['scores'].tolist(),
            model_results['autoencoder']['errors'].tolist(),
            evaluation_results['ensemble_results']['results']
        )
        
        # Build response
        response = {
            "dataset_summary": dataset_summary,
            "performance": evaluation_results['performance'],
            "confusion_matrices": evaluation_results['confusion_matrices'],
            "flagged_transactions": flagged_transactions,
            "anomaly_scores_over_time": anomaly_time_series
        }
        
        return jsonify(create_success_response(response))
        
    except Exception as e:
        return jsonify(create_error_response(f"Server error: {str(e)}", 500)[0]), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify(create_error_response("File too large. Maximum size is 100MB")[0]), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    return jsonify(create_error_response("Internal server error")[0]), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
