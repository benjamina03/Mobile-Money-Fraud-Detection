"""
Streamlit Application for Mobile Money Fraud Detection
Main application with 4 pages: Login, Real-Time Monitor, Batch Analysis, Investigation & Reports
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Import custom modules
from preprocessing import preprocess_data, preprocess_single_transaction, engineer_features, select_features
from models import HybridModel, save_models, load_models


# Page configuration
st.set_page_config(
    page_title="Mobile Money Fraud Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'flagged_transactions' not in st.session_state:
    st.session_state.flagged_transactions = []
if 'anomaly_scores_history' not in st.session_state:
    st.session_state.anomaly_scores_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'hybrid_model' not in st.session_state:
    st.session_state.hybrid_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None


def login_page():
    """Page 1: Login Authentication"""
    st.title("🔒 Mobile Money Fraud Detection System")
    st.subheader("Login")
    
    # Create a centered login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Please enter your credentials")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary", use_container_width=True):
            # Hardcoded credentials for prototype
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.success("✓ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")
        
        st.info("**Demo Credentials:**\n\nUsername: `admin`\n\nPassword: `admin123`")


def load_or_train_models(X_train):
    """Helper function to load or train models"""
    model_path = 'trained_models'
    
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'scaler.pkl')):
        try:
            st.info("Loading pre-trained models...")
            hybrid_model, scaler = load_models(model_path)
            st.session_state.hybrid_model = hybrid_model
            st.session_state.scaler = scaler
            st.session_state.models_loaded = True
            st.success("✓ Models loaded successfully!")
            return hybrid_model, scaler
        except Exception as e:
            st.warning(f"Could not load models: {e}. Training new models...")
    
    # Train new models
    st.info("Training models... This may take a few minutes.")
    progress_bar = st.progress(0)
    
    input_dim = X_train.shape[1]
    hybrid_model = HybridModel(input_dim=input_dim)
    
    progress_bar.progress(25)
    with st.spinner("Training Isolation Forest..."):
        hybrid_model.iso_forest.train(X_train)
    
    progress_bar.progress(50)
    with st.spinner("Training Autoencoder..."):
        hybrid_model.autoencoder.train(X_train, epochs=30)
    
    progress_bar.progress(75)
    with st.spinner("Fitting DBSCAN..."):
        hybrid_model.dbscan.train(X_train)
    
    progress_bar.progress(100)
    st.success("✓ All models trained successfully!")
    
    return hybrid_model, None  # Return None for scaler as it's passed separately


def real_time_monitor():
    """Page 2: Real-Time Transaction Monitor"""
    st.title("📊 Real-Time Transaction Monitor")
    st.markdown("Monitor transactions as they arrive and detect fraud in real-time.")
    
    # File uploader for test dataset
    uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=['csv'], key="realtime_upload")
    
    if uploaded_file is not None:
        # Load and prepare data
        df_test = pd.read_csv(uploaded_file)
        st.info(f"Loaded {len(df_test)} transactions for simulation")
        
        # Prepare models if not already loaded
        if not st.session_state.models_loaded:
            with st.spinner("Preparing models..."):
                # Use a sample for training if models don't exist
                df_train = df_test.sample(min(10000, len(df_test)), random_state=42)
                X_train, scaler, _ = preprocess_data(df_train.copy())
                
                hybrid_model, _ = load_or_train_models(X_train)
                st.session_state.hybrid_model = hybrid_model
                st.session_state.scaler = scaler
                st.session_state.models_loaded = True
        
        # Simulation controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            num_transactions = st.number_input("Transactions to simulate", min_value=1, max_value=100, value=20)
        
        with col2:
            threshold = st.slider("Block Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
        
        # Start simulation button
        if st.button("▶ Start Simulation", type="primary"):
            st.session_state.anomaly_scores_history = []
            st.session_state.flagged_transactions = []
            
            # Create placeholders for live updates
            status_placeholder = st.empty()
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            # Sample transactions
            transactions_to_process = df_test.sample(n=min(num_transactions, len(df_test)))
            
            total_processed = 0
            total_blocked = 0
            
            for idx, (_, transaction) in enumerate(transactions_to_process.iterrows()):
                # Preprocess transaction
                transaction_df = pd.DataFrame([transaction])
                X_transaction, _, df_processed = preprocess_data(
                    transaction_df.copy(), 
                    scaler=st.session_state.scaler
                )
                
                # Get predictions
                predictions, hybrid_scores, individual_scores = st.session_state.hybrid_model.predict(
                    X_transaction, 
                    threshold=threshold
                )
                
                total_processed += 1
                hybrid_score = hybrid_scores[0]
                is_blocked = predictions[0] == 1
                
                if is_blocked:
                    total_blocked += 1
                
                # Store anomaly score
                st.session_state.anomaly_scores_history.append({
                    'transaction_id': idx,
                    'score': hybrid_score,
                    'blocked': is_blocked
                })
                
                # Display status
                with status_placeholder.container():
                    if is_blocked:
                        st.error(f"🚫 **TRANSACTION BLOCKED** - Hybrid Score: {hybrid_score:.3f}")
                        st.markdown(f"""
                        **Transaction Details:**
                        - Type: {transaction.get('type', 'N/A')}
                        - Amount: ${transaction.get('amount', 0):,.2f}
                        - Isolation Forest Score: {individual_scores['isolation_forest'][0]:.3f}
                        - Autoencoder Score: {individual_scores['autoencoder'][0]:.3f}
                        - DBSCAN Score: {individual_scores['dbscan'][0]:.3f}
                        """)
                        
                        # Store flagged transaction
                        flagged_data = transaction.to_dict()
                        flagged_data['hybrid_score'] = hybrid_score
                        flagged_data['reason'] = f"High hybrid score ({hybrid_score:.3f})"
                        if individual_scores['autoencoder'][0] > 0.7:
                            flagged_data['reason'] += ", High reconstruction error"
                        if individual_scores['isolation_forest'][0] > 0.7:
                            flagged_data['reason'] += ", Isolation Forest anomaly"
                        st.session_state.flagged_transactions.append(flagged_data)
                    else:
                        st.success(f"✅ **TRANSACTION APPROVED** - Hybrid Score: {hybrid_score:.3f}")
                        st.markdown(f"""
                        **Transaction Details:**
                        - Type: {transaction.get('type', 'N/A')}
                        - Amount: ${transaction.get('amount', 0):,.2f}
                        """)
                
                # Update metrics
                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Processed", total_processed)
                    m2.metric("Blocked", total_blocked)
                    m3.metric("Approved", total_processed - total_blocked)
                    m4.metric("Block Rate", f"{(total_blocked/total_processed*100):.1f}%")
                
                # Update chart
                if len(st.session_state.anomaly_scores_history) > 0:
                    with chart_placeholder.container():
                        scores_df = pd.DataFrame(st.session_state.anomaly_scores_history)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        
                        colors = ['red' if x else 'green' for x in scores_df['blocked']]
                        ax.scatter(scores_df['transaction_id'], scores_df['score'], c=colors, alpha=0.6)
                        ax.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
                        ax.set_xlabel('Transaction Number')
                        ax.set_ylabel('Hybrid Anomaly Score')
                        ax.set_title('Real-Time Anomaly Scores')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                
                # Simulate delay
                time.sleep(1)
            
            st.success(f"✓ Simulation complete! Processed {total_processed} transactions, blocked {total_blocked}.")


def batch_analysis():
    """Page 3: Batch Analysis"""
    st.title("📁 Batch Analysis")
    st.markdown("Upload a CSV file for bulk fraud detection analysis.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], key="batch_upload")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(df)} transactions")
        
        # Show preview
        with st.expander("Preview Data"):
            st.dataframe(df.head(10))
        
        threshold = st.slider("Anomaly Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05, key="batch_threshold")
        
        if st.button("🔍 Analyze Transactions", type="primary"):
            # Prepare models if needed
            if not st.session_state.models_loaded:
                with st.spinner("Training models..."):
                    df_train = df.sample(min(10000, len(df)), random_state=42)
                    X_train, scaler, _ = preprocess_data(df_train.copy())
                    
                    hybrid_model, _ = load_or_train_models(X_train)
                    st.session_state.hybrid_model = hybrid_model
                    st.session_state.scaler = scaler
                    st.session_state.models_loaded = True
            
            # Process all transactions
            with st.spinner("Processing transactions..."):
                X_processed, _, df_processed = preprocess_data(df.copy(), scaler=st.session_state.scaler)
                predictions, hybrid_scores, individual_scores = st.session_state.hybrid_model.predict(
                    X_processed, 
                    threshold=threshold
                )
            
            # Calculate statistics
            num_anomalies = predictions.sum()
            num_normal = len(predictions) - num_anomalies
            
            # Display metrics
            st.markdown("### Analysis Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                st.metric("Anomalies Detected", num_anomalies)
            with col3:
                st.metric("Normal Transactions", num_normal)
            
            # Visualization
            st.markdown("### Anomaly Distribution")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scatter plot
            colors = ['red' if p == 1 else 'blue' for p in predictions]
            ax1.scatter(range(len(hybrid_scores)), hybrid_scores, c=colors, alpha=0.5, s=20)
            ax1.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
            ax1.set_xlabel('Transaction Index')
            ax1.set_ylabel('Hybrid Anomaly Score')
            ax1.set_title('Anomaly Scores: Normal (Blue) vs Anomalies (Red)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Histogram
            ax2.hist(hybrid_scores[predictions == 0], bins=50, alpha=0.7, label='Normal', color='blue')
            ax2.hist(hybrid_scores[predictions == 1], bins=50, alpha=0.7, label='Anomalies', color='red')
            ax2.axvline(x=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
            ax2.set_xlabel('Hybrid Anomaly Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show anomalous transactions
            if num_anomalies > 0:
                st.markdown("### Detected Anomalies")
                anomaly_indices = np.where(predictions == 1)[0]
                df_anomalies = df.iloc[anomaly_indices].copy()
                df_anomalies['hybrid_score'] = hybrid_scores[anomaly_indices]
                df_anomalies['iso_score'] = individual_scores['isolation_forest'][anomaly_indices]
                df_anomalies['ae_score'] = individual_scores['autoencoder'][anomaly_indices]
                df_anomalies['dbscan_score'] = individual_scores['dbscan'][anomaly_indices]
                
                st.dataframe(df_anomalies)
                
                # Download button
                csv = df_anomalies.to_csv(index=False)
                st.download_button(
                    label="📥 Download Anomalies CSV",
                    data=csv,
                    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def investigation_reports():
    """Page 4: Investigation & Reports"""
    st.title("🔍 Investigation & Reports")
    st.markdown("View flagged transactions and download reports.")
    
    if len(st.session_state.flagged_transactions) == 0:
        st.info("No flagged transactions yet. Run the Real-Time Monitor or Batch Analysis first.")
    else:
        st.success(f"✓ Found {len(st.session_state.flagged_transactions)} flagged transactions")
        
        # Convert to DataFrame
        df_flagged = pd.DataFrame(st.session_state.flagged_transactions)
        
        # Display table
        st.markdown("### Flagged Transactions")
        
        # Select columns to display
        display_cols = ['type', 'amount', 'nameOrig', 'nameDest', 'hybrid_score', 'reason']
        available_cols = [col for col in display_cols if col in df_flagged.columns]
        
        st.dataframe(df_flagged[available_cols], use_container_width=True)
        
        # Statistics
        st.markdown("### Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Flagged", len(df_flagged))
        
        with col2:
            if 'amount' in df_flagged.columns:
                avg_amount = df_flagged['amount'].mean()
                st.metric("Avg Amount", f"${avg_amount:,.2f}")
        
        with col3:
            if 'hybrid_score' in df_flagged.columns:
                avg_score = df_flagged['hybrid_score'].mean()
                st.metric("Avg Score", f"{avg_score:.3f}")
        
        # Visualization
        if 'type' in df_flagged.columns:
            st.markdown("### Transaction Type Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            type_counts = df_flagged['type'].value_counts()
            type_counts.plot(kind='bar', ax=ax, color='coral')
            ax.set_xlabel('Transaction Type')
            ax.set_ylabel('Count')
            ax.set_title('Flagged Transactions by Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Download report
        st.markdown("### Download Report")
        csv = df_flagged.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv,
            file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Clear button
        if st.button("🗑️ Clear Flagged Transactions"):
            st.session_state.flagged_transactions = []
            st.rerun()


def main():
    """Main application logic with navigation"""
    
    # Check login status
    if not st.session_state.logged_in:
        login_page()
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/security-lock.png", width=80)
        st.title("Navigation")
        
        selected = option_menu(
            menu_title=None,
            options=["Real-Time Monitor", "Batch Analysis", "Investigation & Reports"],
            icons=["activity", "folder", "search"],
            menu_icon="cast",
            default_index=0,
        )
        
        st.markdown("---")
        st.markdown("### User Info")
        st.info("👤 **User:** admin")
        
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.flagged_transactions = []
            st.session_state.anomaly_scores_history = []
            st.rerun()
    
    # Route to selected page
    if selected == "Real-Time Monitor":
        real_time_monitor()
    elif selected == "Batch Analysis":
        batch_analysis()
    elif selected == "Investigation & Reports":
        investigation_reports()


if __name__ == "__main__":
    main()
