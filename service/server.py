# service/server.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
from fraud.service import FraudService
from fraud.service.ttypes import FraudPrediction

from service.detector import FraudDetector
from service.eda import EDAReportGenerator
from service.analyzer import FraudAnalyzer

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

# Configure server logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FraudServer")

class FraudServiceHandler:
    def __init__(self):
        logger.info("Initializing fraud detection service handler")
        self.detector = FraudDetector()
        self.eda_generator = EDAReportGenerator()
        self.fraud_analyzer = FraudAnalyzer()
    
    def predictFraud(self, transaction):
        """Predict whether a transaction is fraudulent"""
        logger.info(f"Received fraud prediction request, transaction type: {transaction.type}, amount: {transaction.amount}")
        is_fraud, probability, feature_importance = self.detector.predict(transaction)
        logger.info(f"Completed fraud prediction, result: {'fraud' if is_fraud else 'normal'}, probability: {probability:.4f}")
        return FraudPrediction(isFraud=is_fraud, 
                             fraudProbability=probability,
                             featureImportance=feature_importance)
    
    def trainModel(self, data_path):
        """Train fraud detection model"""
        logger.info(f"Received model training request, data path: {data_path}")
        result = self.detector.train(data_path)
        logger.info(f"Model training completed, result: {'successful' if result else 'failed'}")
        return result
    
    def generateEDAReport(self, request):
        """Generate EDA analysis report"""
        logger.info(f"Received EDA report generation request, data path: {request.dataPath}")
        report = self.eda_generator.generate_report(request)
        report_size = len(report) / 1024 if report else 0
        logger.info(f"EDA report generation completed, size: {report_size:.2f} KB")
        return report
    
    def generateFraudAnalysisReport(self, request):
        """Generate fraud analysis report"""
        logger.info(f"Received fraud analysis report generation request, transaction type: {request.transaction.type}")
        report = self.fraud_analyzer.generate_fraud_analysis(request)
        report_size = len(report) / 1024 if report else 0
        logger.info(f"Fraud analysis report generation completed, size: {report_size:.2f} KB")
        return report

def start_server(port=9090):
    logger.info(f"Preparing to start fraud detection service, port: {port}")
    handler = FraudServiceHandler()
    processor = FraudService.Processor(handler)
    transport = TSocket.TServerSocket(host='0.0.0.0', port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    
    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    
    logger.info(f"Starting fraud detection service, listening on port {port}...")
    server.serve()