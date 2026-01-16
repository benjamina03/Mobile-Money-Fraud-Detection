# Mobile Money Fraud Detection System 🔒

A hybrid unsupervised machine learning system for detecting fraudulent mobile money transactions using Streamlit. This system combines Isolation Forest, Autoencoder, and DBSCAN algorithms to provide real-time fraud detection.

## 🎯 Features

- **Real-Time Transaction Monitoring**: Simulates live transaction processing with immediate fraud detection
- **Batch Analysis**: Upload CSV files for bulk transaction analysis
- **Hybrid ML Approach**: Combines three unsupervised algorithms:
  - Isolation Forest (40% weight)
  - Autoencoder Neural Network (40% weight)
  - DBSCAN Clustering (20% weight)
- **Interactive Dashboard**: User-friendly Streamlit interface with visualizations
- **Investigation Reports**: Track and download flagged transactions

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.9+
- **ML Libraries**: Scikit-learn, PyTorch
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📋 Prerequisites

- Python 3.9 or higher
- pip package manager

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/benjamina03/Mobile-Money-Fraud-Detection.git
   cd Mobile-Money-Fraud-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**
   
   Download the PaySim dataset (or use your own mobile money transaction data) and place it in the project directory. The dataset should have the following columns:
   - `step`: Time step of the transaction
   - `type`: Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)
   - `amount`: Transaction amount
   - `nameOrig`: Customer ID (originator)
   - `oldbalanceOrg`: Initial balance before transaction (originator)
   - `newbalanceOrig`: New balance after transaction (originator)
   - `nameDest`: Recipient ID
   - `oldbalanceDest`: Initial balance before transaction (recipient)
   - `newbalanceDest`: New balance after transaction (recipient)
   
   Optional columns (used for validation but not training):
   - `isFraud`: Actual fraud label
   - `isFlaggedFraud`: System flag

## 🎮 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Login**
   - Username: `admin`
   - Password: `admin123`

3. **Navigate through the application**:

   ### Real-Time Monitor
   - Upload your test dataset
   - Click "Start Simulation" to process transactions one by one
   - Watch real-time fraud detection with visual alerts
   - Adjust the block threshold to tune sensitivity

   ### Batch Analysis
   - Upload a CSV file
   - Analyze all transactions at once
   - View scatter plots and statistics
   - Download detected anomalies

   ### Investigation & Reports
   - Review all flagged transactions
   - View detailed reasons for blocking
   - Download comprehensive reports

## 📊 How It Works

### Feature Engineering
The system automatically creates derived features:
- **Transaction Velocity**: Number of transactions per user per time step
- **Balance Error**: Discrepancy between expected and actual balances
- **Type Encoding**: One-hot encoded transaction types
- **Amount Ratios**: Transaction amount relative to account balance

### Hybrid Scoring
Each transaction receives a hybrid anomaly score:

```
Hybrid_Score = (0.4 × IsoForest_Score) + (0.4 × Autoencoder_Score) + (0.2 × DBSCAN_Score)
```

- **Threshold**: Default is 0.75 (transactions above this are blocked)
- **Scores normalized** to [0, 1] range where 1 = highly anomalous

### Model Training
Models are automatically trained on first use and saved to the `trained_models/` directory for future sessions.

## 📁 Project Structure

```
Mobile-Money-Fraud-Detection/
├── app.py                  # Main Streamlit application
├── models.py               # ML models (Isolation Forest, Autoencoder, DBSCAN)
├── preprocessing.py        # Data preprocessing and feature engineering
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 🔧 Configuration

### Adjust Model Parameters

Edit `models.py` to tune:
- Isolation Forest contamination rate
- Autoencoder architecture and epochs
- DBSCAN eps and min_samples
- Hybrid score weights

### Modify Threshold

In the application, use the slider to adjust the blocking threshold in real-time.

## 🎯 Performance Metrics

The system provides:
- **Real-time alerts**: Immediate feedback on transaction status
- **Visual analytics**: Score distributions and trends
- **Detailed reports**: Transaction-level anomaly breakdowns

## 🔐 Security Notes

- Current version uses hardcoded credentials for demo purposes
- For production use, implement proper authentication
- Store sensitive data securely
- Never commit actual transaction data to version control

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is for educational and demonstration purposes.

## 👨‍💻 Author

Benjamin - [benjamina03](https://github.com/benjamina03)

## 🙏 Acknowledgments

- PaySim synthetic dataset creators
- Streamlit team for the amazing framework
- PyTorch and Scikit-learn communities
