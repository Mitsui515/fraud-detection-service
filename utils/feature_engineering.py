import numpy as np

def extract_features(transaction):
    """Extract features from transaction data"""
    # Create feature list
    features = []
    
    # Basic features
    features.append(transaction.amount)
    features.append(transaction.oldBalanceOrig)
    features.append(transaction.newBalanceOrig)
    features.append(transaction.oldBalanceDest)
    features.append(transaction.newBalanceDest)
    
    # Derived features - balance differences
    balance_diff_orig = transaction.oldBalanceOrig - transaction.newBalanceOrig
    features.append(balance_diff_orig)
    
    balance_diff_dest = transaction.newBalanceDest - transaction.oldBalanceDest
    features.append(balance_diff_dest)
    
    # Transaction type features
    features.append(1 if transaction.type == "TRANSFER" else 0)
    features.append(1 if transaction.type == "CASH_OUT" else 0)
    features.append(1 if transaction.type == "CASH_IN" else 0)
    features.append(1 if transaction.type == "DEBIT" else 0)
    features.append(1 if transaction.type == "PAYMENT" else 0)
    
    # Amount to balance ratio
    features.append(transaction.amount / (transaction.oldBalanceOrig + 1e-6))
    
    # Zero balance flags
    features.append(1 if abs(transaction.oldBalanceOrig) < 1e-6 else 0)
    features.append(1 if abs(transaction.oldBalanceDest) < 1e-6 else 0)
    
    # Post-transaction balance zero flag
    features.append(1 if abs(transaction.newBalanceOrig) < 1e-6 else 0)
    
    # Transaction amount anomaly detection
    features.append(1 if transaction.amount > 10000 else 0)
    
    # Transaction amount relative to destination account balance
    features.append(transaction.amount / (transaction.oldBalanceDest + 1e-6))
    
    # Post-transaction account balance change ratio
    features.append((transaction.newBalanceOrig - transaction.oldBalanceOrig) / (transaction.oldBalanceOrig + 1e-6))
    features.append((transaction.newBalanceDest - transaction.oldBalanceDest) / (transaction.oldBalanceDest + 1e-6))
    
    # Large transaction flag
    features.append(1 if transaction.amount > 1000000 else 0)
    
    # Post-transaction account balance to transaction amount ratio
    features.append(transaction.newBalanceOrig / (transaction.amount + 1e-6))
    
    # Feature name list
    feature_names = [
        'amount', 'oldBalanceOrig', 'newBalanceOrig', 'oldBalanceDest', 'newBalanceDest',
        'balanceDiffOrig', 'balanceDiffDest',
        'isTransfer', 'isCashOut', 'isCashIn', 'isDebit', 'isPayment',
        'amountToBalanceRatio', 'zeroBalanceOrig', 'zeroBalanceDest',
        'newZeroBalanceOrig', 'highAmount',
        'amountToDestBalanceRatio', 'origBalanceChangeRatio', 'destBalanceChangeRatio',
        'isLargeTransaction', 'postTransactionBalanceRatio'
    ]
    
    return np.array(features), feature_names