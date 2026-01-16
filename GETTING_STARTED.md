# Getting Started Guide

## Quick Start

This guide will help you get the Mobile Money Fraud Detection system up and running.

### Prerequisites

- Python 3.9 or higher
- pip package manager
- ~2GB free disk space (for dependencies)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/benjamina03/Mobile-Money-Fraud-Detection.git
   cd Mobile-Money-Fraud-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data** (Optional - for testing)
   ```bash
   python generate_sample_data.py
   ```
   This creates three datasets:
   - `train_data.csv` (10,000 transactions)
   - `test_data.csv` (500 transactions)
   - `demo_data.csv` (50 transactions)

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the web interface**
   - Open your browser to `http://localhost:8501`
   - Login with credentials:
     - Username: `admin`
     - Password: `admin123`

## Using the Application

### Real-Time Monitor

1. Upload a test dataset (CSV file)
2. Wait for models to train (first time only - models are saved for future use)
3. Set the number of transactions to simulate
4. Adjust the block threshold (default: 0.75)
5. Click "Start Simulation"
6. Watch transactions being processed in real-time with color-coded alerts:
   - 🟢 **Green** = Transaction Approved
   - 🔴 **Red** = Transaction Blocked (Fraud Detected)

### Batch Analysis

1. Upload a CSV file with transactions
2. Set the anomaly threshold
3. Click "Analyze Transactions"
4. View results:
   - Total transactions processed
   - Number of anomalies detected
   - Visual scatter plots and histograms
   - Detailed anomaly table
5. Download the anomalies report as CSV

### Investigation & Reports

1. View all flagged transactions from your session
2. See detailed reasons for each flag
3. Download comprehensive reports
4. Clear flagged transactions when done

## Data Format

Your CSV file should contain these columns:

| Column | Description |
|--------|-------------|
| step | Time step (hour) |
| type | Transaction type (CASH-IN, CASH-OUT, PAYMENT, TRANSFER, DEBIT) |
| amount | Transaction amount |
| nameOrig | Customer ID (originator) |
| oldbalanceOrg | Balance before transaction (originator) |
| newbalanceOrig | Balance after transaction (originator) |
| nameDest | Recipient ID |
| oldbalanceDest | Balance before transaction (recipient) |
| newbalanceDest | Balance after transaction (recipient) |

Optional columns (for validation):
- `isFraud` - Actual fraud label (not used in training)
- `isFlaggedFraud` - System flag (not used in training)

## Model Training

The first time you use the system, it will train three models:
1. **Isolation Forest** - Detects outliers based on isolation
2. **Autoencoder** - Neural network that learns normal patterns
3. **DBSCAN** - Clustering algorithm that identifies noise points

Training typically takes 1-3 minutes on the first run. Models are saved to `trained_models/` directory and reused in future sessions.

## Troubleshooting

### Models won't load
- Delete the `trained_models/` directory
- Restart the application
- Models will retrain automatically

### File upload errors
- Ensure your CSV has the required columns
- Check file size (< 200MB recommended)
- Verify CSV format is valid

### Application won't start
- Check if port 8501 is already in use
- Try: `streamlit run app.py --server.port 8502`

### Poor detection accuracy
- Adjust the block threshold (try 0.6-0.8 range)
- Ensure training data is representative
- Use at least 1000 transactions for training

## Performance Tips

- For large datasets (>100K rows), use batch analysis instead of real-time
- Train on a representative sample (10K transactions is usually sufficient)
- Close other browser tabs to improve performance
- Use Chrome or Firefox for best results

## Security Note

⚠️ This is a **prototype/demo system**. For production use:
- Implement proper authentication (not hardcoded credentials)
- Add database integration (currently in-memory)
- Enable HTTPS
- Add user role management
- Implement audit logging
