# service/analyzer.py
import json
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
logger = logging.getLogger("FraudAnalyzer")

class FraudAnalyzer:
    def __init__(self):
        logger.info("Initializing fraud analysis report generator")
        
    def generate_fraud_analysis(self, request):
        """Generate fraud analysis report"""
        analysis_id = f"analysis_{int(time.time() * 1000)}"
        transaction = request.transaction
        prediction = request.prediction
        depth = request.analysisDepth or "detailed"
        
        logger.info(f"[{analysis_id}] Starting to generate fraud analysis report, transaction type: {transaction.type}, amount: {transaction.amount}, analysis depth: {depth}")
        logger.info(f"[{analysis_id}] Transaction prediction result: {'fraud' if prediction.isFraud else 'normal'}, probability: {prediction.fraudProbability:.4f}")
        
        try:
            start_time = time.time()
            
            # Get feature importance, and sort by importance
            feature_importance = prediction.featureImportance or {}
            sorted_features = sorted(feature_importance.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)
            
            if sorted_features:
                logger.debug(f"[{analysis_id}] Extracted {len(sorted_features)} feature importance metrics, most important feature: {sorted_features[0][0]}")
            else:
                logger.warning(f"[{analysis_id}] No feature importance data available")
            
            # Generate report
            logger.info(f"[{analysis_id}] Starting to build Markdown analysis report...")
            report = self._create_markdown_report(
                transaction, 
                prediction, 
                sorted_features,
                depth
            )
            
            # Calculate report size
            report_size = len(report) / 1024  # KB
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"[{analysis_id}] Report generation completed, size: {report_size:.2f} KB, time elapsed: {elapsed:.2f}ms")
            
            return report
            
        except Exception as e:
            logger.error(f"[{analysis_id}] Error generating fraud analysis report: {str(e)}", exc_info=True)
            return f"Error generating fraud analysis report: {str(e)}"
    
    def _create_markdown_report(self, transaction, prediction, feature_importance, depth):
        """Create Markdown formatted fraud analysis report"""
        analysis_id = f"analysis_{int(time.time() * 1000)}"
        logger.debug(f"[{analysis_id}] Starting to build Markdown report content")
        
        report = []
        
        # Report title
        report.append("# Transaction Fraud Analysis Report")
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Transaction summary
        logger.debug(f"[{analysis_id}] Adding transaction summary section")
        report.append("## 1. Transaction Summary")
        report.append(f"* Transaction type: {transaction.type}")
        report.append(f"* Transaction amount: {transaction.amount:.2f}")
        report.append(f"* Originating account: {transaction.nameOrig}")
        report.append(f"* Destination account: {transaction.nameDest}")
        
        # Fraud prediction results
        logger.debug(f"[{analysis_id}] Adding fraud prediction results section")
        report.append("\n## 2. Fraud Prediction Results")
        report.append(f"* Prediction result: {'**Fraudulent transaction**' if prediction.isFraud else 'Normal transaction'}")
        report.append(f"* Fraud probability: {prediction.fraudProbability*100:.2f}%")
        
        # Risk factor analysis
        logger.debug(f"[{analysis_id}] Adding risk factor analysis section")
        report.append("\n## 3. Risk Factor Analysis")
        
        if feature_importance:
            report.append("### 3.1 Main Risk Factors (Sorted by Importance)")
            for feature, importance in feature_importance[:5]:  # Only show top 5 important features
                report.append(f"* **{feature}**: Importance score {importance:.4f}")
                logger.debug(f"[{analysis_id}] Important feature: {feature}, score: {importance:.4f}")
                
            # Explanation of important features
            report.append("\n### 3.2 Risk Factor Explanation")
            for feature, _ in feature_importance[:3]:  # Only explain top 3 important features
                explanation = self._get_feature_explanation(feature, transaction)
                report.append(f"* **{feature}**: {explanation}")
                logger.debug(f"[{analysis_id}] Adding feature explanation: {feature}")
        
        # Transaction-specific analysis based on type
        logger.debug(f"[{analysis_id}] Adding transaction-specific analysis section, transaction type: {transaction.type}")
        report.append("\n## 4. Transaction Specific Analysis")
        if transaction.type == "TRANSFER":
            logger.debug(f"[{analysis_id}] Performing transfer transaction specific analysis")
            report.append(self._analyze_transfer(transaction, prediction))
        elif transaction.type == "CASH_OUT":
            logger.debug(f"[{analysis_id}] Performing cash-out transaction specific analysis")
            report.append(self._analyze_cash_out(transaction, prediction))
        else:
            logger.debug(f"[{analysis_id}] No specific analysis available for transaction type: {transaction.type}")
            report.append(f"No specific risk pattern analysis available for {transaction.type} type transactions.")
        
        # If detailed report, add more analysis
        if depth == "detailed" or depth == "comprehensive":
            logger.debug(f"[{analysis_id}] Adding account behavior analysis (analysis depth: {depth})")
            report.append("\n## 5. Account Behavior Analysis")
            report.append(self._analyze_account_behavior(transaction))
            
            logger.debug(f"[{analysis_id}] Adding amount anomaly analysis")
            report.append("\n## 6. Amount Anomaly Analysis")
            report.append(self._analyze_amount(transaction))
            
        # If comprehensive report, add recommended actions
        if depth == "comprehensive":
            logger.debug(f"[{analysis_id}] Adding recommended actions (analysis depth: {depth})")
            report.append("\n## 7. Recommended Actions")
            report.append(self._get_recommended_actions(transaction, prediction))
        
        # Summary
        logger.debug(f"[{analysis_id}] Adding summary section")
        report.append("\n## 8. Summary")
        if prediction.isFraud:
            report.append("Based on the analysis, this transaction exhibits multiple fraud characteristics. It's recommended to reject the transaction and further investigate.")
        else:
            report.append("Based on the analysis, this transaction has a low risk level with no obvious fraud characteristics.")
        
        logger.debug(f"[{analysis_id}] Markdown report construction completed")
        return "\n".join(report)
    
    def _get_feature_explanation(self, feature, transaction):
        """Get explanation for a feature"""
        explanations = {
            "amount": f"Transaction amount is {transaction.amount:.2f}, " +
                     ("abnormally large, exceeding normal transaction range." if transaction.amount > 10000 else "within normal range."),
            
            "balanceDiffOrig": "Abnormal balance change in the originating account, potentially indicating unusual fund movement.",
            
            "amountToBalanceRatio": "Abnormal ratio of transaction amount to account balance, suggesting possible account compromise.",
            
            "zeroBalanceDest": "Receiving account had zero balance, receiving large funds into newly opened accounts is high-risk behavior.",
            
            "isTransfer": "This is a transfer transaction, requiring attention to fund flow direction.",
            
            "isCashOut": "This is a cash-out transaction, cash withdrawals are common fraud methods."
        }
        
        return explanations.get(feature, "This feature has significant impact on fraud detection.")
    
    def _analyze_transfer(self, transaction, prediction):
        """Analyze transfer transaction"""
        analysis_id = f"transfer_{int(time.time() * 1000)}"
        logger.debug(f"[{analysis_id}] Analyzing transfer transaction, amount: {transaction.amount}")
        
        analysis = []
        
        # Check relationship between transaction amount and account balance
        orig_ratio = transaction.amount / (transaction.oldBalanceOrig + 0.01)
        if orig_ratio > 0.9:
            logger.debug(f"[{analysis_id}] Detected high-risk feature: transfer-to-balance ratio {orig_ratio:.2f}")
            analysis.append("* Transfer amount exceeds 90% of original account balance, which is typical fund-clearing behavior, high risk.")
        
        # Check receiving account status
        if transaction.oldBalanceDest == 0:
            logger.debug(f"[{analysis_id}] Detected high-risk feature: receiving account has zero balance")
            analysis.append("* Receiving account had zero previous balance, new accounts receiving large funds is a fraud risk signal.")
            
        # Check post-transaction balance
        expected_new_balance = transaction.oldBalanceOrig - transaction.amount
        if abs(expected_new_balance - transaction.newBalanceOrig) > 0.01:
            logger.debug(f"[{analysis_id}] Detected high-risk feature: pre/post-transaction balance mismatch")
            analysis.append("* Balance calculation before and after transaction doesn't match, suggesting hidden transactions or system anomalies.")
            
        if not analysis:
            logger.debug(f"[{analysis_id}] No high-risk features detected")
            analysis.append("* This transfer transaction doesn't exhibit obvious risk features.")
            
        return "\n".join(analysis)
    
    def _analyze_cash_out(self, transaction, prediction):
        """Analyze cash-out transaction"""
        analysis_id = f"cashout_{int(time.time() * 1000)}"
        logger.debug(f"[{analysis_id}] Analyzing cash-out transaction, amount: {transaction.amount}")
        
        analysis = []
        
        # Check cash-out amount
        if transaction.amount > 5000:
            logger.debug(f"[{analysis_id}] Detected high-risk feature: large cash-out ({transaction.amount})")
            analysis.append("* Large cash-out (>5000) carries high risk, especially shortly after account opening.")
            
        # Check post-cash-out balance
        if transaction.newBalanceOrig < 0.1 * transaction.oldBalanceOrig:
            logger.debug(f"[{analysis_id}] Detected high-risk feature: significant balance reduction after cash-out")
            analysis.append("* Post-cash-out account balance decreased by over 90%, potentially fund-clearing behavior.")
            
        if not analysis:
            logger.debug(f"[{analysis_id}] No high-risk features detected")
            analysis.append("* This cash-out transaction doesn't show obvious risk features.")
            
        return "\n".join(analysis)
    
    def _analyze_account_behavior(self, transaction):
        """Analyze account behavior"""
        analysis_id = f"account_{int(time.time() * 1000)}"
        logger.debug(f"[{analysis_id}] Analyzing account behavior")
        
        analysis = []
        
        # Originating account analysis
        if transaction.oldBalanceOrig == transaction.newBalanceOrig and transaction.amount > 0:
            logger.debug(f"[{analysis_id}] Detected anomaly: transaction amount greater than 0 but balance unchanged")
            analysis.append("* Anomaly: Transaction amount is greater than 0 but originating account balance is unchanged, suggesting transaction record anomaly.")
        
        # Receiving account analysis
        expected_new_balance_dest = transaction.oldBalanceDest + transaction.amount
        if abs(expected_new_balance_dest - transaction.newBalanceDest) > 0.01:
            logger.debug(f"[{analysis_id}] Detected anomaly: receiving account balance change doesn't match transaction amount")
            analysis.append("* Anomaly: Receiving account's balance change doesn't match the transaction amount, suggesting unclear fund destination.")
            
        if not analysis:
            logger.debug(f"[{analysis_id}] No obvious anomalies found in account behavior")
        
        return "\n".join(analysis) if analysis else "No obvious anomalies found in account behavior analysis."
    
    def _analyze_amount(self, transaction):
        """Analyze transaction amount"""
        analysis_id = f"amount_{int(time.time() * 1000)}"
        logger.debug(f"[{analysis_id}] Analyzing transaction amount: {transaction.amount}")
        
        if transaction.amount > 100000:
            logger.debug(f"[{analysis_id}] Determined as abnormally large amount transaction")
            return "Transaction amount is abnormally large, far exceeding regular transactions, high risk."
        elif transaction.amount > 10000:
            logger.debug(f"[{analysis_id}] Determined as large amount transaction")
            return "Transaction amount is large, within a range requiring attention."
        else:
            logger.debug(f"[{analysis_id}] Determined as normal amount transaction")
            return "Transaction amount is within normal range."
    
    def _get_recommended_actions(self, transaction, prediction):
        """Get recommended actions"""
        analysis_id = f"actions_{int(time.time() * 1000)}"
        logger.debug(f"[{analysis_id}] Generating action recommendations, fraud probability: {prediction.fraudProbability}")
        
        actions = []
        
        if prediction.isFraud and prediction.fraudProbability > 0.8:
            logger.debug(f"[{analysis_id}] Recommending high-risk actions (high probability fraud)")
            actions.append("* **Reject transaction immediately**: This transaction has a high probability of fraud, recommend immediate rejection and freezing related accounts.")
            actions.append("* **Notify risk control team**: Mark this transaction as a high-risk case, arrange for manual review.")
            actions.append("* **Contact account holder**: Attempt to contact the originating account holder to confirm transaction authenticity.")
        elif prediction.isFraud:
            logger.debug(f"[{analysis_id}] Recommending medium-risk actions (possible fraud)")
            actions.append("* **Suspend transaction**: Temporarily hold this transaction pending further verification.")
            actions.append("* **Secondary verification**: Require secondary identity verification from the transaction initiator.")
            actions.append("* **Restrict account**: Temporarily limit large transaction capabilities for related accounts.")
        else:
            logger.debug(f"[{analysis_id}] Recommending low-risk actions (normal transaction)")
            actions.append("* **Normal processing**: This transaction has low risk and can be processed normally.")
            actions.append("* **Record monitoring**: Include the transaction in routine monitoring.")
            
        return "\n".join(actions)