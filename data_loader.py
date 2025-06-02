import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
from typing import Dict, List, Tuple

def load_transaction_data(csv_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load transaction data from CSV and prepare it for ArangoDB.
    Returns tuple of (transactions, users)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Generate unique IDs for transactions and users
    df['TransactionId'] = df.apply(lambda x: hashlib.sha256(
        f"{x['timestamp']}_{x['amount']}_{x['sender']}_{x['receiver']}".encode()
    ).hexdigest()[:16], axis=1)
    
    df['SenderId'] = df['sender'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16])
    df['ReceiverId'] = df['receiver'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16])
    
    # Convert timestamp to Unix timestamp if not already
    if df['timestamp'].dtype == 'object':
        df['Timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9
    else:
        df['Timestamp'] = df['timestamp']
    
    # Prepare transaction documents
    transactions = []
    for _, row in df.iterrows():
        transaction = {
            "_key": row['TransactionId'],
            "TransactionId": row['TransactionId'],
            "Timestamp": int(row['Timestamp']),
            "Amount": float(row['amount']),
            "SenderId": row['SenderId'],
            "ReceiverId": row['ReceiverId'],
            "TransactionType": row.get('type', 'TRANSFER'),
            "Status": "COMPLETED",
            "FraudScore": 0.0,
            "RiskScore": 0.0
        }
        transactions.append(transaction)
    
    # Prepare user documents
    users = set()
    user_stats = {}
    
    for tx in transactions:
        for user_id in [tx['SenderId'], tx['ReceiverId']]:
            if user_id not in users:
                users.add(user_id)
                user_stats[user_id] = {
                    "TransactionCount": 0,
                    "TotalAmount": 0.0,
                    "LastActivity": 0
                }
            
            user_stats[user_id]["TransactionCount"] += 1
            user_stats[user_id]["TotalAmount"] += tx["Amount"]
            user_stats[user_id]["LastActivity"] = max(
                user_stats[user_id]["LastActivity"],
                tx["Timestamp"]
            )
    
    user_docs = []
    for user_id in users:
        stats = user_stats[user_id]
        user_doc = {
            "_key": user_id,
            "UserId": user_id,
            "RiskScore": 0.0,
            "TransactionCount": stats["TransactionCount"],
            "TotalAmount": stats["TotalAmount"],
            "LastActivity": stats["LastActivity"]
        }
        user_docs.append(user_doc)
    
    return transactions, user_docs

def calculate_initial_risk_scores(transactions: List[Dict], users: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Calculate initial risk scores for transactions and users based on basic heuristics.
    """
    # Create user lookup for faster access
    user_lookup = {u["UserId"]: u for u in users}
    
    # Calculate transaction statistics
    amounts = [tx["Amount"] for tx in transactions]
    mean_amount = np.mean(amounts)
    std_amount = np.std(amounts)
    
    # Update transaction risk scores
    for tx in transactions:
        # Calculate z-score for amount
        z_score = (tx["Amount"] - mean_amount) / std_amount if std_amount > 0 else 0
        
        # Basic risk factors
        risk_factors = [
            abs(z_score) if abs(z_score) > 3 else 0,  # Unusual amount
            0.5 if tx["Amount"] > 10000 else 0,  # Large transaction
        ]
        
        # Combine risk factors
        tx["RiskScore"] = min(sum(risk_factors) / len(risk_factors), 1.0)
    
    # Update user risk scores based on their transactions
    for user in users:
        user_txs = [
            tx for tx in transactions
            if tx["SenderId"] == user["UserId"] or tx["ReceiverId"] == user["UserId"]
        ]
        
        if user_txs:
            # Average transaction risk score
            avg_tx_risk = np.mean([tx["RiskScore"] for tx in user_txs])
            
            # Transaction frequency risk
            tx_frequency = len(user_txs) / (max(user["LastActivity"] - min(tx["Timestamp"] for tx in user_txs), 1) / 86400)
            freq_risk = min(tx_frequency / 10, 1.0)  # Cap at 10 transactions per day
            
            # Combine risk factors
            user["RiskScore"] = min((avg_tx_risk + freq_risk) / 2, 1.0)
    
    return transactions, users 