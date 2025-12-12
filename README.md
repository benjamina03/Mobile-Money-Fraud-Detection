# Mobile Money Fraud Detection System

A lightweight hybrid unsupervised machine learning prototype for detecting fraudulent mobile money transactions. Built with Streamlit for rapid prototyping, using scikit-learn and PyTorch for ML models.

## Features

### 🤖 Machine Learning Models
- **Isolation Forest** - Tree-based anomaly detection (scikit-learn)
- **Local Outlier Factor (LOF)** - Density-based outlier detection (scikit-learn)
- **Deep Autoencoder** - Neural network reconstruction error analysis (PyTorch)
- **Hybrid Ensemble** - Weighted voting combining all three models

### 💻 Streamlit Dashboard
- Interactive file upload for CSV datasets
- Real-time model training and analysis
- Plotly-powered visualizations:
  - Confusion matrix comparison (grouped bar chart)
  - Model F1 score comparison (donut chart)
  - Anomaly scores over time (area chart)
- Downloadable flagged transactions report
- Dark theme with neon accent styling

## Project Structure

```
Mobile-Money-Fraud-Detection/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
└── backend/
    ├── __init__.py
    ├── preprocessing.py   # Data cleaning and feature engineering
    ├── models.py          # ML model implementations (PyTorch + scikit-learn)
    ├── evaluation.py      # Metrics and ensemble logic
    └── utils.py           # Utility functions
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

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser (usually opens automatically at `http://localhost:8501`)

3. Upload a PaySim-format CSV file using the sidebar with the following columns:
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

4. View real-time analysis results and download flagged transactions

## Technology Stack

- **UI Framework**: Streamlit
- **Machine Learning**: scikit-learn, PyTorch
- **Data Processing**: pandas, NumPy
- **Visualization**: Plotly

## License

MIT License
