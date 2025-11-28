# Mobile Money Fraud Detection System

A full-stack hybrid unsupervised machine learning platform for detecting fraudulent mobile money transactions. This system uses three complementary anomaly detection algorithms (Isolation Forest, Local Outlier Factor, and Deep Autoencoder) combined in an ensemble approach for robust fraud detection.

## Features

### 🔐 Backend (Python + Flask)
- **Isolation Forest** - Tree-based anomaly detection
- **Local Outlier Factor (LOF)** - Density-based outlier detection
- **Deep Autoencoder** - Neural network-based reconstruction error analysis
- **Hybrid Ensemble** - Weighted voting combining all three models
- RESTful API with `/analyze` and `/health` endpoints
- Support for PaySim-format mobile money datasets
- Comprehensive preprocessing and feature engineering

### 💻 Frontend (HTML/CSS/JS)
- Modern dark-themed cyber-style dashboard
- Drag-and-drop CSV file upload
- Real-time analysis with loading indicators
- Interactive Chart.js visualizations:
  - Confusion matrix comparison (bar chart)
  - Model performance comparison (donut chart)
  - Anomaly scores over time (line chart)
- Responsive design with neon accent styling
- Flagged transactions table with risk scores

## Project Structure

```
Mobile-Money-Fraud-Detection/
├── app.py                 # Flask application entry point
├── requirements.txt       # Python dependencies
├── backend/
│   ├── __init__.py
│   ├── preprocessing.py   # Data cleaning and feature engineering
│   ├── models.py          # ML model implementations
│   ├── evaluation.py      # Metrics and ensemble logic
│   └── utils.py           # Utility functions
└── frontend/
    ├── index.html         # Main dashboard page
    ├── styles.css         # Custom CSS with neon accents
    └── app.js             # Frontend JavaScript logic
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/benjamina03/Mobile-Money-Fraud-Detection.git
cd Mobile-Money-Fraud-Detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload a PaySim-format CSV file with the following columns:
   - `step` - Time step
   - `type` - Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
   - `amount` - Transaction amount
   - `nameOrig` - Origin account
   - `oldbalanceOrg` - Origin balance before transaction
   - `newbalanceOrig` - Origin balance after transaction
   - `nameDest` - Destination account
   - `oldbalanceDest` - Destination balance before transaction
   - `newbalanceDest` - Destination balance after transaction
   - `isFraud` (optional) - Ground truth fraud label

4. Click "Analyze" to process the dataset

## API Endpoints

### POST /analyze
Upload a CSV file for fraud analysis.

**Response:**
```json
{
  "success": true,
  "dataset_summary": {
    "total_transactions": 10000,
    "fraudulent_transactions": 120,
    "valid_transactions": 9880,
    "fraud_rate": "1.20%"
  },
  "performance": {
    "isolation_forest": { "accuracy": 0.98, "precision": 0.89, "recall": 0.91, "f1": 0.90 },
    "lof": { "accuracy": 0.97, "precision": 0.83, "recall": 0.88, "f1": 0.85 },
    "autoencoder": { "reconstruction_error": 0.04, "threshold": 0.08, "loss": 0.05 },
    "hybrid": { "score": 0.975, "accuracy": 0.98, "precision": 0.91, "f1": 0.92 }
  },
  "confusion_matrices": {
    "IF": { "TN": 9800, "FP": 80, "FN": 10, "TP": 110 },
    "LOF": { "TN": 9750, "FP": 130, "FN": 15, "TP": 105 },
    "AE": { "TN": 9780, "FP": 100, "FN": 12, "TP": 108 }
  },
  "flagged_transactions": [
    {
      "id": 3241,
      "amount": 2800.00,
      "if_score": 0.91,
      "lof_score": 0.88,
      "ae_score": 0.95,
      "final_result": "Fraud"
    }
  ],
  "anomaly_scores_over_time": [...]
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "Backend running",
  "version": "1.0.0"
}
```

## Technology Stack

- **Backend**: Python, Flask, Flask-CORS
- **Machine Learning**: scikit-learn, TensorFlow/Keras
- **Data Processing**: pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Chart.js

## License

MIT License
