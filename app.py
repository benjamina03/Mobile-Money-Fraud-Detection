"""
Streamlit Application for Mobile Money Fraud Detection System.
A lightweight prototype using hybrid unsupervised ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Import backend modules
from backend.preprocessing import preprocess_dataset, get_dataset_summary, load_and_clean_dataset
from backend.models import train_all_models
from backend.evaluation import evaluate_all_models, get_anomaly_scores_over_time
from backend.utils import validate_dataset_columns, format_flagged_transactions

# Page configuration
st.set_page_config(
    page_title="Mobile Money Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with neon accents
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-purple: #7C3AED;
        --primary-indigo: #6366F1;
        --primary-cyan: #06B6D4;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #7C3AED 0%, #6366F1 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #A0A0B2;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    .metric-label {
        color: #A0A0B2;
        font-size: 0.9rem;
    }
    
    /* Success/Error badges */
    .fraud-badge {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .valid-badge {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Mobile Money Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Hybrid Unsupervised Machine Learning for Secure Transactions</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Configuration")
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV Dataset",
            type=['csv'],
            help="Upload a PaySim-format CSV file for fraud analysis"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Required Columns")
        st.markdown("""
        - `step` - Time step
        - `type` - Transaction type
        - `amount` - Transaction amount
        - `nameOrig` - Origin account
        - `oldbalanceOrg` - Origin balance before
        - `newbalanceOrig` - Origin balance after
        - `nameDest` - Destination account
        - `oldbalanceDest` - Dest balance before
        - `newbalanceDest` - Dest balance after
        """)
        
        st.markdown("---")
        st.markdown("### 🤖 Models Used")
        st.markdown("""
        1. **Isolation Forest** - Tree-based anomaly detection
        2. **Local Outlier Factor** - Density-based detection
        3. **Autoencoder (PyTorch)** - Neural network reconstruction
        """)
    
    # Main content
    if uploaded_file is not None:
        # Process the file
        with st.spinner("🔄 Processing dataset..."):
            try:
                # Load and validate
                file_content = io.BytesIO(uploaded_file.read())
                df_raw = load_and_clean_dataset(file_content)
                
                # Validate columns
                is_valid, error_msg = validate_dataset_columns(df_raw)
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                    return
                
                # Check for ground truth labels
                has_labels = 'isFraud' in df_raw.columns
                original_labels = df_raw['isFraud'].values if has_labels else None
                
                # Reset and preprocess
                file_content.seek(0)
                X, df_processed, feature_columns = preprocess_dataset(file_content)
                
                # Determine contamination
                if has_labels:
                    contamination = max(0.01, min(0.5, np.mean(original_labels)))
                else:
                    contamination = 0.1
                
                st.success("✅ Dataset loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
                return
        
        # Train models
        with st.spinner("🧠 Training ML models..."):
            try:
                model_results = train_all_models(X, contamination=contamination)
                evaluation_results = evaluate_all_models(model_results, original_labels)
                st.success("✅ Models trained successfully!")
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
                return
        
        # Get results
        final_predictions = evaluation_results['ensemble_results']['predictions']
        dataset_summary = get_dataset_summary(df_processed, final_predictions)
        
        # Display Dataset Summary
        st.markdown("---")
        st.header("📊 Dataset Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Transactions",
                value=f"{dataset_summary['total_transactions']:,}"
            )
        
        with col2:
            st.metric(
                label="Fraudulent Transactions",
                value=f"{dataset_summary['fraudulent_transactions']:,}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Valid Transactions",
                value=f"{dataset_summary['valid_transactions']:,}"
            )
        
        with col4:
            st.metric(
                label="Fraud Rate",
                value=dataset_summary['fraud_rate']
            )
        
        # Model Performance
        st.markdown("---")
        st.header("🎯 Model Performance Metrics")
        
        performance = evaluation_results['performance']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader("🌲 Isolation Forest")
            st.metric("Accuracy", f"{performance['isolation_forest']['accuracy']:.2%}")
            st.metric("Precision", f"{performance['isolation_forest']['precision']:.2%}")
            st.metric("Recall", f"{performance['isolation_forest']['recall']:.2%}")
            st.metric("F1 Score", f"{performance['isolation_forest']['f1']:.2%}")
        
        with col2:
            st.subheader("📍 Local Outlier Factor")
            st.metric("Accuracy", f"{performance['lof']['accuracy']:.2%}")
            st.metric("Precision", f"{performance['lof']['precision']:.2%}")
            st.metric("Recall", f"{performance['lof']['recall']:.2%}")
            st.metric("F1 Score", f"{performance['lof']['f1']:.2%}")
        
        with col3:
            st.subheader("🔄 Autoencoder")
            st.metric("Reconstruction Error", f"{performance['autoencoder']['reconstruction_error']:.4f}")
            st.metric("Threshold", f"{performance['autoencoder']['threshold']:.4f}")
            st.metric("Loss", f"{performance['autoencoder']['loss']:.4f}")
        
        with col4:
            st.subheader("🔗 Hybrid Ensemble")
            st.metric("Ensemble Score", f"{performance['hybrid']['score']:.4f}")
            st.metric("Accuracy", f"{performance['hybrid']['accuracy']:.2%}")
            st.metric("Precision", f"{performance['hybrid']['precision']:.2%}")
            st.metric("F1 Score", f"{performance['hybrid']['f1']:.2%}")
        
        # Confusion Matrices
        st.markdown("---")
        st.header("📈 Graphical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix Comparison
            cm = evaluation_results['confusion_matrices']
            
            fig_cm = go.Figure(data=[
                go.Bar(name='Isolation Forest', x=['TN', 'FP', 'FN', 'TP'], 
                       y=[cm['IF']['TN'], cm['IF']['FP'], cm['IF']['FN'], cm['IF']['TP']],
                       marker_color='#7C3AED'),
                go.Bar(name='LOF', x=['TN', 'FP', 'FN', 'TP'],
                       y=[cm['LOF']['TN'], cm['LOF']['FP'], cm['LOF']['FN'], cm['LOF']['TP']],
                       marker_color='#6366F1'),
                go.Bar(name='Autoencoder', x=['TN', 'FP', 'FN', 'TP'],
                       y=[cm['AE']['TN'], cm['AE']['FP'], cm['AE']['FN'], cm['AE']['TP']],
                       marker_color='#06B6D4')
            ])
            
            fig_cm.update_layout(
                title='Confusion Matrix Comparison',
                barmode='group',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Model Performance Donut Chart
            f1_scores = [
                performance['isolation_forest']['f1'],
                performance['lof']['f1'],
                performance['hybrid']['f1']
            ]
            
            fig_perf = go.Figure(data=[go.Pie(
                labels=['Isolation Forest', 'LOF', 'Hybrid Ensemble'],
                values=[score * 100 for score in f1_scores],
                hole=0.4,
                marker_colors=['#7C3AED', '#6366F1', '#06B6D4']
            )])
            
            fig_perf.update_layout(
                title='Model F1 Score Comparison',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        
        # Anomaly Scores Over Time
        step_values = df_processed['step'].values
        time_series = get_anomaly_scores_over_time(model_results, step_values)
        
        if time_series:
            fig_time = go.Figure()
            
            steps = [d['step'] for d in time_series]
            
            fig_time.add_trace(go.Scatter(
                x=steps, y=[d['if_avg'] for d in time_series],
                mode='lines', name='Isolation Forest',
                line=dict(color='#7C3AED', width=2),
                fill='tozeroy', fillcolor='rgba(124, 58, 237, 0.1)'
            ))
            
            fig_time.add_trace(go.Scatter(
                x=steps, y=[d['lof_avg'] for d in time_series],
                mode='lines', name='LOF',
                line=dict(color='#6366F1', width=2),
                fill='tozeroy', fillcolor='rgba(99, 102, 241, 0.1)'
            ))
            
            fig_time.add_trace(go.Scatter(
                x=steps, y=[d['ae_avg'] for d in time_series],
                mode='lines', name='Autoencoder',
                line=dict(color='#06B6D4', width=2),
                fill='tozeroy', fillcolor='rgba(6, 182, 212, 0.1)'
            ))
            
            fig_time.update_layout(
                title='Anomaly Scores Over Time',
                xaxis_title='Time Step',
                yaxis_title='Anomaly Score',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Flagged Transactions Table
        st.markdown("---")
        st.header("🚨 Flagged Transactions")
        
        flagged = format_flagged_transactions(
            df_processed,
            model_results['isolation_forest']['scores'].tolist(),
            model_results['lof']['scores'].tolist(),
            model_results['autoencoder']['errors'].tolist(),
            evaluation_results['ensemble_results']['results']
        )
        
        if flagged:
            df_flagged = pd.DataFrame(flagged)
            df_flagged.columns = ['Transaction ID', 'Amount ($)', 'IF Score', 'LOF Score', 'AE Score', 'Result']
            
            # Format the dataframe
            st.dataframe(
                df_flagged.style.format({
                    'Amount ($)': '${:,.2f}',
                    'IF Score': '{:.4f}',
                    'LOF Score': '{:.4f}',
                    'AE Score': '{:.4f}'
                }).map(
                    lambda x: 'color: #EF4444' if x == 'Fraud' else 'color: #10B981',
                    subset=['Result']
                ),
                use_container_width=True,
                height=400
            )
            
            # Download button for flagged transactions
            csv = df_flagged.to_csv(index=False)
            st.download_button(
                label="📥 Download Flagged Transactions",
                data=csv,
                file_name="flagged_transactions.csv",
                mime="text/csv"
            )
        else:
            st.info("No fraudulent transactions detected.")
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%); border-radius: 16px; border: 1px solid rgba(124, 58, 237, 0.3);">
                <h2 style="color: #FFFFFF;">👆 Upload a CSV file to get started</h2>
                <p style="color: #A0A0B2;">Use the sidebar to upload your PaySim-format dataset for fraud analysis.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #6B6B80;">Mobile Money Fraud Detection System © 2024 | '
        'Powered by Hybrid Unsupervised Machine Learning</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
