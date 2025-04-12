#!/usr/bin/env python
import os
import argparse
from service.server import start_server

def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Service')
    parser.add_argument('--port', type=int, default=9090, help='Service port')
    parser.add_argument('--train', type=str, help='CSV file path for training')
    
    args = parser.parse_args()
    
    if args.train and os.path.exists(args.train):
        from service.detector import FraudDetector
        detector = FraudDetector()
        success = detector.train(args.train)
        if success:
            print(f"Successfully trained model using {args.train}")
        else:
            print(f"Model training failed")
    
    start_server(args.port)

if __name__ == '__main__':
    main()