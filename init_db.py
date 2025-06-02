import os
import pandas as pd
from database.arango_client import arango_db_client
import csv

def load_transaction_data():
    data_file = os.path.join('data', 'bank_transactions_data_2.csv')
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        return None
    return pd.read_csv(data_file)

def init_database():
    print("Initializing database...")
    
    # Read transactions from CSV
    transactions = []
    with open('data/bank_transactions_data_2.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map CSV columns to our schema
            transaction = {
                'TransactionId': row['TransactionID'],
                'timestamp': row['TransactionDate'],
                'Amount': float(row['TransactionAmount']),
                'AccountID': row['AccountID'],
                'MerchantID': row['MerchantID'],
                'Type': row['TransactionType'],
                'Location': row['Location'],
                'Channel': row['Channel'],
                'DeviceID': row['DeviceID'],
                'IP': row['IP Address'],
                'CustomerAge': int(row['CustomerAge']),
                'CustomerOccupation': row['CustomerOccupation'],
                'TransactionDuration': int(row['TransactionDuration']),
                'LoginAttempts': int(row['LoginAttempts']),
                'AccountBalance': float(row['AccountBalance']),
                'PreviousTransactionDate': row['PreviousTransactionDate']
            }
            transactions.append(transaction)
    
    # Add users and transactions to database
    db_client = arango_db_client
    
    # Create a set of unique users from transactions
    users = {}
    for tx in transactions:
        if tx['AccountID'] not in users:
            users[tx['AccountID']] = {
                'UserId': tx['AccountID'],
                'Age': tx['CustomerAge'],
                'Occupation': tx['CustomerOccupation'],
                'LastKnownBalance': tx['AccountBalance']
            }
    
    # Add users to database
    for user in users.values():
        db_client.add_user(user)
    
    # Add transactions to database
    for tx in transactions:
        db_client.add_transaction(tx, tx['AccountID'], tx['MerchantID'])
    
    print(f"Loaded {len(transactions)} transactions")

if __name__ == "__main__":
    init_database() 