import networkx as nx
import matplotlib.pyplot as plt
from database.arango_client import arango_db_client
from fraud_detection.fraud_analyzer import FraudAnalyzer
import matplotlib.colors as mcolors
from typing import Dict, Any
import numpy as np

class TransactionGraphVisualizer:
    def __init__(self):
        self.db_client = arango_db_client
        self.fraud_analyzer = FraudAnalyzer()
        
    def create_graph(self):
        """Create a NetworkX graph from transaction data."""
        G = nx.DiGraph()
        
        # Get all transactions and analyze them
        transactions = self.db_client.get_all_transactions()
        
        # Create a color map for risk scores
        color_map = plt.cm.RdYlGn_r  # Red for high risk, yellow for medium, green for low
        
        # Add nodes and edges
        for tx in transactions:
            # Analyze transaction
            analysis = self.fraud_analyzer.analyze_transaction(tx)
            risk_score = analysis['risk_score']
            
            # Add account node if it doesn't exist
            account_id = tx.get('AccountID')
            if not G.has_node(account_id):
                G.add_node(account_id, 
                          node_type='account',
                          label=f"Account\n{account_id}",
                          size=1000)
            
            # Add merchant node if it doesn't exist
            merchant_id = tx.get('MerchantID')
            if not G.has_node(merchant_id):
                G.add_node(merchant_id, 
                          node_type='merchant',
                          label=f"Merchant\n{merchant_id}",
                          size=1000)
            
            # Add transaction node
            tx_id = tx.get('TransactionId')
            amount = float(tx.get('Amount', 0))
            G.add_node(tx_id,
                      node_type='transaction',
                      label=f"${amount:.2f}",
                      size=500,
                      risk_score=risk_score,
                      color=color_map(risk_score))
            
            # Add edges
            G.add_edge(account_id, tx_id, weight=amount)
            G.add_edge(tx_id, merchant_id, weight=amount)
        
        return G
    
    def visualize(self, output_file: str = None):
        """Visualize the transaction graph."""
        G = self.create_graph()
        
        # Create figure and axes
        fig = plt.figure(figsize=(20, 20))
        
        # Create a larger subplot for the graph and a smaller one for the colorbar
        gs = fig.add_gridspec(1, 20)  # 1 row, 20 columns
        ax_graph = fig.add_subplot(gs[0, :-1])  # Use all but the last column
        ax_cbar = fig.add_subplot(gs[0, -1])    # Use the last column for colorbar
        
        # Set up the layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw different node types
        node_types = {'account': 'lightblue', 'merchant': 'lightgreen', 'transaction': 'white'}
        for node_type, color in node_types.items():
            nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == node_type]
            sizes = [G.nodes[n].get('size', 300) for n in nodes]
            
            if node_type == 'transaction':
                # Color transactions by risk score
                colors = [G.nodes[n].get('color', 'white') for n in nodes]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=sizes,
                                     node_color=colors, alpha=0.8, ax=ax_graph)
            else:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=sizes,
                                     node_color=color, alpha=0.6, ax=ax_graph)
        
        # Draw edges
        edge_weights = [G.edges[e].get('weight', 1.0) for e in G.edges()]
        max_weight = max(edge_weights)
        normalized_weights = [w/max_weight * 2 for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.3,
                             edge_color='gray', arrows=True, arrowsize=10, ax=ax_graph)
        
        # Add labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax_graph)
        
        # Add title
        ax_graph.set_title("Transaction Network\nRed nodes indicate high-risk transactions", 
                          fontsize=16, pad=20)
        
        # Add colorbar for risk scores
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, cax=ax_cbar, label='Risk Score')
        ax_cbar.set_ylabel('Risk Score', fontsize=12)
        
        # Remove axes
        ax_graph.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Graph visualization saved to {output_file}")
        else:
            plt.show()
        
        # Close the figure to free memory
        plt.close()
        
        # Print some graph statistics
        print("\nGraph Statistics:")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Number of accounts: {len([n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'account'])}")
        print(f"Number of merchants: {len([n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'merchant'])}")
        print(f"Number of transactions: {len([n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'transaction'])}")

def main():
    # Install required packages if not already installed
    try:
        import networkx
        import matplotlib
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "networkx", "matplotlib"])
    
    visualizer = TransactionGraphVisualizer()
    
    # Create visualizations
    print("Generating transaction graph visualization...")
    visualizer.visualize("transaction_graph.png")
    
    print("\nVisualization complete! Check transaction_graph.png for the output.")

if __name__ == "__main__":
    main() 