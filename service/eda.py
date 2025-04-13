import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EDAReportGenerator")

class EDAReportGenerator:
    def __init__(self):
        logger.info("Initializing EDA report generator")
        
    def generate_report(self, request):
        """Generate EDA report"""
        job_id = f"eda_{int(time.time())}"
        logger.info(f"[{job_id}] Starting to generate EDA report, data path: {request.dataPath}")
        
        try:
            start_time = time.time()
            
            # Load data
            logger.info(f"[{job_id}] Loading data file...")
            df = pd.read_csv(request.dataPath)
            logger.info(f"[{job_id}] Data loaded successfully, total {len(df)} rows, {df.shape[1]} columns")
            
            # Filter by date range if specified
            if request.startDate and request.endDate and 'timestamp' in df.columns:
                logger.info(f"[{job_id}] Applying date filter: {request.startDate} to {request.endDate}")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                original_count = len(df)
                df = df[(df['timestamp'] >= request.startDate) & 
                         (df['timestamp'] <= request.endDate)]
                logger.info(f"[{job_id}] {len(df)} rows remaining after date filter (before: {original_count})")
            
            # Generate report
            logger.info(f"[{job_id}] Starting to generate Markdown report...")
            report = self._create_markdown_report(df, request.focusFeatures)
            
            # Calculate report size
            report_size = len(report) / 1024  # KB
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"[{job_id}] Report generation completed, size: {report_size:.2f} KB, time elapsed: {elapsed:.2f}ms")
            
            return report
            
        except Exception as e:
            logger.error(f"[{job_id}] Error generating EDA report: {str(e)}", exc_info=True)
            return f"Error generating EDA report: {str(e)}"
    
    def _create_markdown_report(self, df, focus_features=None):
        """Create Markdown formatted EDA report"""
        logger.debug("Starting to build Markdown report content")
        
        report = []
        
        # Report title
        report.append("# Transaction Data Exploratory Analysis Report")
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Data overview
        logger.debug("Adding data overview section")
        report.append("## 1. Data Overview")
        report.append(f"* Number of records: {len(df)}")
        report.append(f"* Number of features: {df.shape[1]}")
        if 'isFraud' in df.columns:
            fraud_rate = df['isFraud'].mean() * 100
            report.append(f"* Fraud rate: {fraud_rate:.2f}%")
            logger.debug(f"Data fraud rate: {fraud_rate:.2f}%")
            
        # Data structure
        logger.debug("Adding data structure section")
        report.append("\n## 2. Data Structure")
        report.append("### 2.1 Data Types")
        dtypes = df.dtypes.astype(str)
        report.append("```")
        for col, dtype in zip(dtypes.index, dtypes.values):
            report.append(f"{col:<20} {dtype}")
        report.append("```")
        
        # Statistical summary
        logger.debug("Adding statistical summary section")
        report.append("\n### 2.2 Statistical Summary")
        report.append("```")
        report.append(df.describe().to_string())
        report.append("```")
        
        # Missing values analysis
        logger.debug("Adding missing values analysis section")
        report.append("\n## 3. Missing Values Analysis")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.debug(f"Found missing values, total: {missing.sum()}")
            report.append("Column               Missing Count  Missing Percentage")
            report.append("-----------------------------------")
            for col in missing.index:
                if missing[col] > 0:
                    report.append(f"{col:<20} {missing[col]:>10} {missing[col]/len(df)*100:>10.2f}%")
        else:
            logger.debug("No missing values found")
            report.append("No missing values in the data.")
            
        # Transaction type analysis
        if 'type' in df.columns:
            logger.debug("Adding transaction type analysis section")
            report.append("\n## 4. Transaction Type Analysis")
            type_counts = df['type'].value_counts()
            report.append("Transaction Type     Count    Percentage")
            report.append("-----------------------------------")
            for idx, count in enumerate(type_counts):
                tx_type = type_counts.index[idx]
                report.append(f"{tx_type:<20} {count:>8} {count/len(df)*100:>8.2f}%")
                
        # Fraud distribution
        if 'isFraud' in df.columns:
            logger.debug("Adding fraud distribution analysis section")
            report.append("\n## 5. Fraud Distribution")
            fraud_df = df[df['isFraud'] == 1]
            non_fraud_df = df[df['isFraud'] == 0]
            
            report.append(f"* Fraudulent transactions: {len(fraud_df)} ({len(fraud_df)/len(df)*100:.2f}%)")
            report.append(f"* Normal transactions: {len(non_fraud_df)} ({len(non_fraud_df)/len(df)*100:.2f}%)")
            
            # Fraud rate by transaction type
            if 'type' in df.columns:
                logger.debug("Adding fraud rate by transaction type analysis")
                report.append("\n### 5.1 Fraud Rate by Transaction Type")
                report.append("Transaction Type     Fraud Count  Fraud Rate")
                report.append("-----------------------------------")
                for tx_type in df['type'].unique():
                    type_df = df[df['type'] == tx_type]
                    fraud_count = type_df['isFraud'].sum()
                    fraud_rate = fraud_count / len(type_df) * 100 if len(type_df) > 0 else 0
                    report.append(f"{tx_type:<20} {fraud_count:>8} {fraud_rate:>8.2f}%")
                    logger.debug(f"Fraud rate for transaction type {tx_type}: {fraud_rate:.2f}%")
                
        # Specific feature analysis
        if focus_features:
            logger.debug(f"Adding focus features analysis, features: {focus_features}")
            report.append("\n## 6. Focus Features Analysis")
            for feature in focus_features:
                if feature in df.columns:
                    logger.debug(f"Analyzing feature: {feature}")
                    report.append(f"\n### 6.1 {feature} Feature Analysis")
                    # Basic statistics
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        report.append("Statistical Summary:")
                        report.append("```")
                        report.append(df[feature].describe().to_string())
                        report.append("```")
                    else:
                        value_counts = df[feature].value_counts().head(10)
                        report.append("Top 10 Most Common Values:")
                        report.append("```")
                        for val, count in zip(value_counts.index, value_counts.values):
                            report.append(f"{val}: {count} ({count/len(df)*100:.2f}%)")
                        report.append("```")
        
        # Summary and recommendations
        logger.debug("Adding summary and recommendations section")
        report.append("\n## 7. Summary and Recommendations")
        report.append("* Data quality assessment: Based on the missing values analysis, the data quality is good.")
        if 'isFraud' in df.columns:
            fraud_rate = df['isFraud'].mean() * 100
            report.append(f"* The fraud rate in this dataset is {fraud_rate:.2f}%.")
            report.append("* Recommendation: Enhance feature engineering and model monitoring for high-risk transaction types.")
        
        logger.debug("Markdown report construction completed")
        return "\n".join(report)