# Fraud Detection Service

This is the model service of our Master Capstone Project.

## Environment Setup and Running

### 1. Environment Setup

It is recommended to use a virtual environment to manage dependencies.

```bash
# Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Compile Thrift Definitions (If API needs modification)

If you modify the `transaction.thrift` file, you need to regenerate the Python code.

```bash
# Ensure the thrift compiler is installed
thrift -r --gen py -out . api/thrift/transaction.thrift
```

The generated code will be located in the `api/thrift/` directory.

*Note: The current repository already includes pre-generated Thrift code (`fraud/service/ttypes.py`, `fraud/service/FraudService.py`), so this step is not necessary unless you modify the `.thrift` file.*

### 3. Run the Fraud Detection Service

Use `main.py` to start the Thrift server. By default, it will listen on port 9090.

```bash
python main.py
```

You should see log output similar to the following, indicating the server has started successfully:

```
INFO:FraudServer:Starting fraud detection service, listening on port 9090...
```

### 4. Run the Client (Example)

You can run `client.py` to test the service. It will connect to the locally running service, send a sample transaction, and print the prediction results and analysis report.

```bash
python client.py
```

### 5. (Optional) Train the Model

If you have new training data (in CSV format), you can retrain the model using the `--train` parameter.

```bash
python main.py --train /path/to/your/training_data.csv
```

After training is complete, the service will automatically use the newly trained model. The model files will be saved in the `models` directory (`models/fraud_model.pkl`, `models/feature_importance.pkl`).
