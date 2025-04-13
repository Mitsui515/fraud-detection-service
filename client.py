from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from fraud.service import FraudService
from fraud.service.ttypes import TransactionData, FraudAnalysisRequest, EDARequest

transport = TSocket.TSocket('localhost', 9090)
transport = TTransport.TBufferedTransport(transport)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = FraudService.Client(protocol)

transport.open()

tx = TransactionData(
    type="TRANSFER",
    amount=50000.0,
    nameOrig="C12345",
    oldBalanceOrig=100000.0,
    newBalanceOrig=50000.0,
    nameDest="C67890",
    oldBalanceDest=10000.0,
    newBalanceDest=60000.0
)

result = client.predictFraud(tx)
print(result)
print(f"Fraud prediction: {'Yes' if result.isFraud else 'No'}, Probability: {result.fraudProbability:.2f}")

fraud_request = FraudAnalysisRequest(
    transaction=tx,
    prediction=result,
)
report = client.generateFraudAnalysisReport(fraud_request)
print("\nFraud Analysis Report:")
print(report)

eda_request = EDARequest(
    dataPath="/mnt/e/HKU Courses/Project/data/transactions.csv",
)
eda_report = client.generateEDAReport(eda_request)
print("\nEDA Analysis Report:")
print(eda_report)

transport.close()