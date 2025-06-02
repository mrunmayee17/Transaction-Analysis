import os
from arango import ArangoClient
from arango.exceptions import GraphCreateError, CollectionCreateError, DatabaseCreateError, DocumentInsertError
from dotenv import load_dotenv

load_dotenv()

class ArangoDBClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ArangoDBClient, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # Load configuration from environment
        self.host = os.getenv("ARANGODB_HOST", "http://localhost:8529")
        self.username = os.getenv("ARANGODB_USER", "root")
        self.password = os.getenv("ARANGODB_PASSWORD", "")
        self.db_name = os.getenv("ARANGODB_DB_NAME", "fraud_detection")

        # Initialize the client
        self.client = ArangoClient(hosts=self.host)

        try:
            # First connect to _system database without authentication
            sys_db = self.client.db("_system")
            print("Connected to _system database")

            # Create database if it doesn't exist
            if not sys_db.has_database(self.db_name):
                sys_db.create_database(self.db_name)
                print(f"Created database '{self.db_name}'")

            # Now connect to our database without authentication
            self.db = self.client.db(self.db_name)
            print(f"Connected to database '{self.db_name}'")

        except Exception as e:
            print(f"Database connection/creation failed: {e}")
            raise

        # Initialize graph components
        self.graph_name = "FraudDetectionGraph"
        self.user_collection_name = "Users"
        self.transaction_collection_name = "Transactions"
        self.sends_edge_collection_name = "Sends"
        self.receives_edge_collection_name = "Receives"
        
        self._init_database()
        self._initialized = True
        print(f"Successfully initialized database '{self.db_name}'")

    def _init_database(self):
        # Create collections if they don't exist
        if not self.db.has_collection(self.user_collection_name):
            self.db.create_collection(self.user_collection_name)
            print(f"Created collection '{self.user_collection_name}'")
        
        if not self.db.has_collection(self.transaction_collection_name):
            self.db.create_collection(self.transaction_collection_name)
            print(f"Created collection '{self.transaction_collection_name}'")

        # Create edge collections if they don't exist
        if not self.db.has_collection(self.sends_edge_collection_name):
            self.db.create_collection(self.sends_edge_collection_name, edge=True)
            print(f"Created edge collection '{self.sends_edge_collection_name}'")
            
        if not self.db.has_collection(self.receives_edge_collection_name):
            self.db.create_collection(self.receives_edge_collection_name, edge=True)
            print(f"Created edge collection '{self.receives_edge_collection_name}'")

        # Create graph if it doesn't exist
        if not self.db.has_graph(self.graph_name):
            edge_definitions = [
                {
                    "edge_collection": self.sends_edge_collection_name,
                    "from_vertex_collections": [self.user_collection_name],
                    "to_vertex_collections": [self.transaction_collection_name]
                },
                {
                    "edge_collection": self.receives_edge_collection_name,
                    "from_vertex_collections": [self.user_collection_name],
                    "to_vertex_collections": [self.transaction_collection_name]
                }
            ]
            
            try:
                self.db.create_graph(self.graph_name, edge_definitions=edge_definitions)
                print(f"Created graph '{self.graph_name}'")
            except (GraphCreateError, CollectionCreateError) as e:
                print(f"Note: {e}")

        # Get references to collections through the graph
        self.graph = self.db.graph(self.graph_name)
        self.users = self.graph.vertex_collection(self.user_collection_name)
        self.transactions = self.graph.vertex_collection(self.transaction_collection_name)
        self.sends_edges = self.graph.edge_collection(self.sends_edge_collection_name)
        self.receives_edges = self.graph.edge_collection(self.receives_edge_collection_name)

    def add_user(self, user_data: dict):
        try:
            # Try to get existing user
            user_id = user_data["UserId"]
            try:
                existing_user = self.users.get(user_id)
                if existing_user:
                    # Update existing user
                    user_data["_key"] = user_id
                    return self.users.update(user_data)
            except:
                pass
            
            # Insert new user
            user_data["_key"] = user_id
            return self.users.insert(user_data)
        except DocumentInsertError as e:
            if "unique constraint violated" in str(e):
                # If it's a duplicate, try to update instead
                return self.users.update(user_data)
            raise

    def add_transaction(self, transaction_data: dict, sender_id: str, receiver_id: str):
        try:
            # Set the transaction key
            transaction_data["_key"] = transaction_data["TransactionId"]
            
            # Try to insert the transaction
            tx_doc = self.transactions.insert(transaction_data)
            
            sender_key = f"{self.user_collection_name}/{sender_id}"
            receiver_key = f"{self.user_collection_name}/{receiver_id}"
            tx_key = tx_doc["_id"]

            # Create 'Sends' edge
            try:
                self.sends_edges.insert({
                    "_from": sender_key,
                    "_to": tx_key,
                    "_key": f"s_{transaction_data['TransactionId']}"
                })
            except DocumentInsertError as e:
                if "unique constraint violated" not in str(e):
                    raise
            
            # Create 'Receives' edge
            try:
                self.receives_edges.insert({
                    "_from": receiver_key,
                    "_to": tx_key,
                    "_key": f"r_{transaction_data['TransactionId']}"
                })
            except DocumentInsertError as e:
                if "unique constraint violated" not in str(e):
                    raise
            
            return tx_doc
        except DocumentInsertError as e:
            if "unique constraint violated" in str(e):
                # If it's a duplicate, try to update instead
                return self.transactions.update(transaction_data)
            raise

    def get_user_transaction_history(self, user_id: str):
        history_query = f"""
            LET sent_tx = (
                FOR tx IN 1..1 OUTBOUND '{self.user_collection_name}/{user_id}' {self.sends_edge_collection_name}
                RETURN tx
            )
            LET received_tx = (
                FOR tx IN 1..1 OUTBOUND '{self.user_collection_name}/{user_id}' {self.receives_edge_collection_name}
                RETURN tx
            )
            RETURN {{ sent: sent_tx, received: received_tx }}
        """
        cursor = self.db.aql.execute(history_query)
        return [doc for doc in cursor]

    def get_statistics(self):
        # Example: Total number of users and transactions
        num_users = self.users.count()
        num_transactions = self.transactions.count()
        return {
            "total_users": num_users,
            "total_transactions": num_transactions,
        }

    def get_transaction_neighbors(self, transaction_id: str, depth: int = 1):
        # This requires a graph traversal query (AQL)
        aql_query = f"""
        FOR v, e, p IN 1..{depth} ANY '{self.transaction_collection_name}/{transaction_id}' GRAPH '{self.graph_name}'
          RETURN {{vertex: v, edge: e, path: p}}
        """
        cursor = self.db.aql.execute(aql_query)
        return [doc for doc in cursor]

    def update_fraud_score(self, transaction_id: str, new_score: float):
        tx_doc = self.transactions.get(transaction_id)
        if tx_doc:
            tx_doc['FraudScore'] = new_score
            self.transactions.update(tx_doc)
            return tx_doc
        return None

    def get_high_risk_users(self, risk_threshold: float):
        # Requires AQL query to filter users by risk score
        aql_query = f"""
        FOR u IN {self.user_collection_name}
          FILTER u.RiskScore >= {risk_threshold}
          RETURN u
        """
        cursor = self.db.aql.execute(aql_query)
        return [doc for doc in cursor]
        
    def get_all_transactions(self):
        return [tx for tx in self.transactions.all()]

# Singleton instance
arango_db_client = ArangoDBClient() 