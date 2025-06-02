from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Transaction(BaseModel):
    TransactionId: str = Field(..., description="Unique identifier for the transaction")
    Timestamp: int = Field(..., description="Unix timestamp of the transaction")
    Amount: float = Field(..., description="Transaction amount")
    SenderId: str = Field(..., description="ID of the sender")
    ReceiverId: str = Field(..., description="ID of the receiver")
    SenderEmail: Optional[str] = Field(None, description="Email of the sender")
    ReceiverEmail: Optional[str] = Field(None, description="Email of the receiver")
    TransactionType: str = Field(..., description="Type of transaction")
    Status: str = Field("PENDING", description="Transaction status")
    FraudScore: float = Field(0.0, description="Calculated fraud score")
    RiskScore: float = Field(0.0, description="Risk score")

class UserInfo(BaseModel):
    UserId: str = Field(..., description="Unique identifier for the user")
    Email: Optional[str] = Field(None, description="User's email")
    RiskScore: float = Field(0.0, description="User's risk score")
    TransactionCount: int = Field(0, description="Number of transactions")
    LastActivity: Optional[int] = Field(None, description="Last activity timestamp")

class TransactionResponse(BaseModel):
    transaction: Transaction
    is_fraudulent: bool
    confidence_score: float
    risk_factors: List[str]
    recommendation: str

class TransactionStats(BaseModel):
    total_transactions: int
    fraud_detected: int
    total_amount: float
    average_fraud_score: float
    high_risk_users: List[UserInfo] 