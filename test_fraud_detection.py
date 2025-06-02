from fraud_detection.fraud_analyzer import FraudAnalyzer
from tabulate import tabulate
from datetime import datetime

def format_amount(amount):
    try:
        return f"${float(amount):,.2f}"
    except (ValueError, TypeError):
        return str(amount)

def main():
    print("Starting fraud detection analysis...")
    
    # Initialize the fraud analyzer
    analyzer = FraudAnalyzer()
    
    # Analyze all transactions
    suspicious_transactions = analyzer.analyze_all_transactions()
    
    # Sort by risk score
    suspicious_transactions.sort(key=lambda x: x['analysis']['risk_score'], reverse=True)
    
    # Prepare data for display
    table_data = []
    for tx in suspicious_transactions:
        table_data.append([
            tx.get('TransactionId'),
            tx.get('timestamp', ''),
            format_amount(tx.get('Amount')),
            tx.get('AccountID', ''),
            tx.get('MerchantID', ''),
            f"{tx['analysis']['risk_score']:.2f}",
            '\n'.join(tx['analysis']['risk_details'])
        ])
    
    # Display results
    headers = ['Transaction ID', 'Timestamp', 'Amount', 'Account ID', 'Merchant ID', 'Risk Score', 'Risk Details']
    print(f"\nSuspicious Transactions Found: {len(suspicious_transactions)}")
    print("\nDetailed Analysis:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total suspicious transactions: {len(suspicious_transactions)}")
    
    # Calculate risk score distribution
    high_risk = len([tx for tx in suspicious_transactions if tx['analysis']['risk_score'] > 0.8])
    medium_risk = len([tx for tx in suspicious_transactions if 0.5 < tx['analysis']['risk_score'] <= 0.8])
    
    print(f"High risk transactions (score > 0.8): {high_risk}")
    print(f"Medium risk transactions (0.5 < score <= 0.8): {medium_risk}")
    
    if suspicious_transactions:
        # Print top 3 highest risk transactions
        print("\nTop 3 Highest Risk Transactions:")
        for i, tx in enumerate(suspicious_transactions[:3], 1):
            print(f"\n{i}. Transaction ID: {tx.get('TransactionId')}")
            print(f"   Timestamp: {tx.get('timestamp', '')}")
            print(f"   Amount: {format_amount(tx.get('Amount'))}")
            print(f"   Risk Score: {tx['analysis']['risk_score']:.2f}")
            print(f"   Risk Factors:")
            for detail in tx['analysis']['risk_details']:
                print(f"   - {detail}")

if __name__ == "__main__":
    main() 