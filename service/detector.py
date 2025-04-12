import logging
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import time

from utils.feature_engineering import extract_features
from fraud.service.ttypes import TransactionData

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FraudDetector")

class FraudDetector:
    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "fraud_model.pkl")
        self.feature_importance_path = os.path.join(model_dir, "feature_importance.pkl")
        logger.info(f"Initializing fraud detector, model directory: {model_dir}")
        self.model = self._load_model()
        self.feature_names = self._load_feature_names()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                logger.info(f"Loading model: {self.model_path}")
                model = joblib.load(self.model_path)
                logger.info(f"Successfully loaded model: {type(model).__name__}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}", exc_info=True)
        else:
            logger.warning(f"Model file does not exist: {self.model_path}")
        return None
    
    def _load_feature_names(self):
        if os.path.exists(self.feature_importance_path):
            try:
                logger.info(f"Loading feature names: {self.feature_importance_path}")
                names = joblib.load(self.feature_importance_path)
                logger.info(f"Successfully loaded feature names, total {len(names)} features")
                return names
            except Exception as e:
                logger.error(f"Failed to load feature names: {e}", exc_info=True)
        else:
            logger.warning(f"Feature importance file does not exist: {self.feature_importance_path}")
        return None
        
    def predict(self, transaction: TransactionData):
        request_id = f"req_{int(time.time() * 1000)}"
        logger.info(f"[{request_id}] Starting fraud prediction, transaction type: {transaction.type}, amount: {transaction.amount}")
        
        if self.model is None:
            logger.warning(f"[{request_id}] Model not loaded, using fallback rules for prediction")
            is_fraud = transaction.amount > 100000
            logger.info(f"[{request_id}] Fallback rule prediction result: {'fraud' if is_fraud else 'normal'}, probability: {0.95 if is_fraud else 0.05}")
            return is_fraud, 0.95 if is_fraud else 0.05, {}
            
        try:
            start_time = time.time()
            features, feature_names = extract_features(transaction)
            logger.debug(f"[{request_id}] Feature extraction completed, total {len(features)} features")
            
            fraud_prob = self.model.predict_proba([features])[0, 1]
            is_fraud = fraud_prob > 0.5
            
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_') and self.feature_names:
                for name, importance in zip(self.feature_names, self.model.feature_importances_):
                    feature_importance[name] = float(importance)
                logger.debug(f"[{request_id}] Extracted {len(feature_importance)} feature importances")
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"[{request_id}] Prediction completed: {'fraud' if is_fraud else 'normal'}, probability: {fraud_prob:.4f}, time elapsed: {elapsed:.2f}ms")
            
            return is_fraud, float(fraud_prob), feature_importance
        except Exception as e:
            logger.error(f"[{request_id}] Prediction error: {e}", exc_info=True)
            is_fraud = transaction.amount > 100000
            logger.info(f"[{request_id}] Using fallback rule after error, result: {'fraud' if is_fraud else 'normal'}")
            return is_fraud, 0.95 if is_fraud else 0.05, {}
    
    def train(self, data_path):
        job_id = f"train_{int(time.time())}"
        logger.info(f"[{job_id}] Starting model training, data path: {data_path}")
        
        try:
            start_time = time.time()
            
            os.makedirs(self.model_dir, exist_ok=True)
            logger.info(f"[{job_id}] Ensuring model directory exists: {self.model_dir}")
            
            logger.info(f"[{job_id}] Reading CSV data: {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"[{job_id}] Data loaded successfully, total {len(df)} rows")
            
            logger.info(f"[{job_id}] Starting feature engineering...")
            features, feature_names = self._prepare_training_features(df)
            logger.info(f"[{job_id}] Feature engineering completed, generated {features.shape[1]} features")
            
            if 'isFraud' not in df.columns:
                logger.error(f"[{job_id}] Data is missing required 'isFraud' column")
                return False
                
            labels = df['isFraud']
            fraud_count = labels.sum()
            logger.info(f"[{job_id}] Preparing training labels, fraud samples: {fraud_count}, fraud rate: {fraud_count/len(df)*100:.2f}%")
            
            logger.info(f"[{job_id}] Starting model training...")
            model = DecisionTreeClassifier(random_state=42)
            model.fit(features, labels)
            logger.info(f"[{job_id}] Model training completed")
            
            logger.info(f"[{job_id}] Saving model to: {self.model_path}")
            joblib.dump(model, self.model_path)
            
            logger.info(f"[{job_id}] Saving feature names to: {self.feature_importance_path}")
            joblib.dump(feature_names, self.feature_importance_path)
            
            self.model = model
            self.feature_names = feature_names
            
            elapsed = time.time() - start_time
            logger.info(f"[{job_id}] Model training and saving successful, time used: {elapsed:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"[{job_id}] Error training model: {e}", exc_info=True)
            return False