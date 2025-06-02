from datetime import datetime, timedelta
from typing import List, Dict, Any
from database.arango_client import arango_db_client

class FraudAnalyzer:
    def __init__(self):
        self.db_client = arango_db_client
        
        # Fraud detection thresholds
        self.AMOUNT_THRESHOLD = 1000  # Transactions above this amount are suspicious
        self.FREQUENCY_THRESHOLD = 3    # More than this many transactions in TIME_WINDOW is suspicious
        self.TIME_WINDOW = timedelta(hours=24)  # Time window for frequency analysis
        self.RISK_SCORE_THRESHOLD = 0.5  # Transactions with risk score above this are suspicious
        
    def analyze_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single transaction for fraud indicators."""
        risk_factors = []
        risk_details = []
        risk_score = 0.0
        
        try:
            # Check transaction amount
            amount = transaction.get('Amount')
            if amount:
                amount_float = float(amount)
                if amount_float > self.AMOUNT_THRESHOLD:
                    risk_score_add = min(0.3 * (amount_float / self.AMOUNT_THRESHOLD), 0.4)
                    risk_factors.append('High amount transaction')
                    risk_details.append(f'Amount ${amount_float:.2f} exceeds threshold ${self.AMOUNT_THRESHOLD:.2f}')
                    risk_score += risk_score_add
            
            # Get transaction history for sender
            account_id = transaction.get('AccountID')
            if account_id:
                sender_history = self.db_client.get_user_transaction_history(account_id)
                if sender_history and sender_history[0].get('sent'):
                    timestamp = transaction.get('timestamp')
                    if timestamp:
                        # Check transaction frequency
                        recent_transactions = self._get_recent_transactions(sender_history[0]['sent'], timestamp)
                        tx_count = len(recent_transactions)
                        if tx_count > self.FREQUENCY_THRESHOLD:
                            risk_score_add = min(0.2 * (tx_count / self.FREQUENCY_THRESHOLD), 0.3)
                            risk_factors.append('High frequency of transactions')
                            risk_details.append(f'{tx_count} transactions in the last {self.TIME_WINDOW.total_seconds()/3600:.0f} hours')
                            risk_score += risk_score_add
                        
                        # Check for unusual patterns
                        pattern_details = self._has_unusual_patterns(transaction, recent_transactions)
                        if pattern_details['is_suspicious']:
                            risk_factors.extend(pattern_details['risk_factors'])
                            risk_details.extend(pattern_details['details'])
                            risk_score += pattern_details['risk_score']
        except Exception as e:
            print(f"Warning: Error analyzing transaction {transaction.get('TransactionId')}: {str(e)}")
            risk_factors.append('Analysis error occurred')
            risk_details.append(str(e))
            
        return {
            'transaction_id': transaction.get('TransactionId'),
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'risk_details': risk_details,
            'is_suspicious': risk_score > self.RISK_SCORE_THRESHOLD
        }
        
    def analyze_all_transactions(self) -> List[Dict[str, Any]]:
        """Analyze all transactions in the database."""
        suspicious_transactions = []
        all_transactions = self.db_client.get_all_transactions()
        
        print(f"Analyzing {len(all_transactions)} transactions...")
        
        for transaction in all_transactions:
            analysis = self.analyze_transaction(transaction)
            if analysis['is_suspicious']:
                suspicious_transactions.append({
                    **transaction,
                    'analysis': analysis
                })
                
        return suspicious_transactions
        
    def _get_recent_transactions(self, transactions: List[Dict[str, Any]], current_time: str) -> List[Dict[str, Any]]:
        """Get transactions within the time window."""
        if not current_time or not transactions:
            return []
            
        try:
            current_dt = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
            window_start = current_dt - self.TIME_WINDOW
            
            recent = []
            for tx in transactions:
                tx_time = tx.get('timestamp')
                if tx_time:
                    try:
                        tx_dt = datetime.strptime(tx_time, '%Y-%m-%d %H:%M:%S')
                        if tx_dt > window_start:
                            recent.append(tx)
                    except ValueError:
                        continue
            return recent
        except ValueError:
            return []
        
    def _has_unusual_patterns(self, current_tx: Dict[str, Any], recent_transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for unusual patterns in transactions."""
        result = {
            'is_suspicious': False,
            'risk_factors': [],
            'details': [],
            'risk_score': 0.0
        }
        
        if not recent_transactions:
            return result
            
        try:
            # Calculate average transaction amount
            amounts = []
            for tx in recent_transactions:
                amount = tx.get('Amount')
                if amount:
                    try:
                        amounts.append(float(amount))
                    except (ValueError, TypeError):
                        continue
                        
            if not amounts:
                return result
                
            avg_amount = sum(amounts) / len(amounts)
            
            # Check if current transaction amount is significantly higher than average
            current_amount = current_tx.get('Amount')
            if current_amount:
                try:
                    current_amount = float(current_amount)
                    amount_ratio = current_amount / avg_amount
                    if amount_ratio > 3:  # 3x higher than average is suspicious
                        result['is_suspicious'] = True
                        result['risk_factors'].append('Unusual transaction amount')
                        result['details'].append(f'Amount ${current_amount:.2f} is {amount_ratio:.1f}x higher than average (${avg_amount:.2f})')
                        result['risk_score'] += min(0.2 * (amount_ratio / 3), 0.3)
                except (ValueError, TypeError):
                    pass
            
            # Check for repeated transactions to same merchant
            merchant_counts = {}
            for tx in recent_transactions:
                merchant = tx.get('MerchantID')
                if merchant:
                    merchant_counts[merchant] = merchant_counts.get(merchant, 0) + 1
            
            current_merchant = current_tx.get('MerchantID')
            if current_merchant:
                merchant_count = merchant_counts.get(current_merchant, 0)
                if merchant_count > 3:  # More than 3 transactions to same merchant is suspicious
                    result['is_suspicious'] = True
                    result['risk_factors'].append('Repeated merchant transactions')
                    result['details'].append(f'{merchant_count} transactions to merchant {current_merchant} in {self.TIME_WINDOW.total_seconds()/3600:.0f} hours')
                    result['risk_score'] += min(0.2 * (merchant_count / 3), 0.3)
                
        except Exception as e:
            print(f"Warning: Error checking patterns: {str(e)}")
            
        return result 