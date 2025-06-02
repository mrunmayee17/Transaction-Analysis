import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit.components.v1 as components
from database.arango_client import arango_db_client
from fraud_detection.fraud_analyzer import FraudAnalyzer
import json
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Set page config to wide mode
st.set_page_config(layout="wide")

class InteractiveGraphApp:
    def __init__(self):
        self.db_client = arango_db_client
        self.fraud_analyzer = FraudAnalyzer()
        self.transactions = None
        self.G = None
        self.setup_llm()
        
    def setup_llm(self):
        """Setup LLM for Q&A functionality."""
        try:
            # Get API key from environment
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            # Initialize Gemini LLM with optimized configuration
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                google_api_key=google_api_key,
                temperature=0.3,
                max_output_tokens=8192,  # Increased for longer responses
                top_p=0.95,
                top_k=40
            )
            
            # Initialize conversation memory with correct keys
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="input",
                output_key="output",
                return_messages=True
            )
            
            # Define the chat prompt template with structured output format
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial fraud detection expert analyzing transaction networks. 
                Analyze the provided transaction data and network metrics to identify patterns, anomalies, and potential fraud indicators.
                
                Focus on:
                1. Transaction patterns and frequency analysis
                2. Network structure and relationships
                3. Risk indicators and suspicious behavior
                4. Temporal patterns and clustering
                
                Format your responses in a clear, structured way with sections and bullet points.
                Always provide specific examples from the data when relevant.
                Keep your responses focused and concise while ensuring all key information is included."""),
                ("human", """Context:
{context}

Network Analysis:
{graph_info}

Question: {input}

Please provide a comprehensive analysis with the following sections:

1. Direct Findings
- Key statistics and metrics


2. Notable Patterns & Anomalies
- Unusual transaction patterns
- Network structure anomalies


3. Risk Indicators
- High-risk transactions
- Suspicious patterns
- Network vulnerabilities
- Unusual behavior patterns

4. Recommendations
- Areas for further investigation
- Specific accounts/merchants to monitor
- Suggested preventive measures
- Data collection improvements"""),
            ])
            
            # Create the chain with proper configuration
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                memory=self.memory,
                verbose=True,
                output_key="output"
            )
            
            st.sidebar.success("✅ LLM initialized successfully")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error initializing LLM: {str(e)}")
            st.sidebar.info("Please ensure GOOGLE_API_KEY is properly set in your .env file")
            raise

    def get_graph_insights(self, G):
        """Extract insights from the transaction graph."""
        insights = {
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'components': nx.number_strongly_connected_components(G),
            'cycles': len(list(nx.simple_cycles(G))),
            'high_degree_nodes': sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5],
            'centrality': {
                'degree': nx.degree_centrality(G),
                'betweenness': nx.betweenness_centrality(G),
                'eigenvector': nx.eigenvector_centrality(G, max_iter=300)
            }
        }
        return insights

    def process_qa(self, question, filtered_transactions):
        """Process Q&A using Gemini LLM with graph-based context."""
        try:
            # Create graph for analysis
            G = nx.DiGraph()
            for _, tx in filtered_transactions.iterrows():
                G.add_edge(tx['AccountID'], tx['MerchantID'], 
                          weight=tx['Amount'],
                          risk_score=tx['risk_score'])

            # Get graph insights
            graph_insights = self.get_graph_insights(G)
            
            # Get pattern analysis
            patterns, _ = self.analyze_transaction_patterns(filtered_transactions)
            
            # Prepare pattern analysis context
            pattern_info = f"""
            Suspicious Pattern Analysis:
            
            1. Circular Transaction Patterns:
            - Number of circular patterns: {len(patterns['circular_patterns'])}
            - Details: {', '.join([f"Pattern {i+1}: {' → '.join(p['nodes'])}" for i, p in enumerate(patterns['circular_patterns'][:3])])}
            
            2. Unusual Amount Transactions:
            - Count: {len(patterns['unusual_amounts'])}
            - Top unusual amounts: {', '.join([f"${tx['Amount']:,.2f} (ID: {tx['TransactionId']})" for tx in patterns['unusual_amounts'][:121]])}
            
            3. High-Frequency Trading:
            - Count: {len(patterns['high_frequency'])}
            - Recent high-frequency transactions: {', '.join([f"ID: {tx['TransactionId']} at {tx['timestamp']}" for tx in patterns['high_frequency'][:968]])}
            
            4. Network Centrality:
            - Most Central (Degree): {sorted(patterns['central_nodes']['degree'].items(), key=lambda x: x[1], reverse=True)[:5]}
            - Key Intermediaries (Betweenness): {sorted(patterns['central_nodes']['betweenness'].items(), key=lambda x: x[1], reverse=True)[:5]}
            - Most Influential (Eigenvector): {sorted(patterns['central_nodes']['eigenvector'].items(), key=lambda x: x[1], reverse=True)[:5]}
            """
            
            # Prepare graph information context
            graph_info = f"""
            Network Metrics:
            - Graph Density: {graph_insights['density']:.3f} (indicates how interconnected the network is)
            - Average Clustering: {graph_insights['avg_clustering']:.3f} (shows tendency to form clusters)
            - Strong Components: {graph_insights['components']} (groups of connected transactions)
            - Cycles Detected: {graph_insights['cycles']} (potential circular transaction patterns)
            
            Most Active Nodes (by connections):
            {', '.join([f"{node}: {degree} transactions" for node, degree in graph_insights['high_degree_nodes']])}
            
            Key Network Players:
            - Most Connected: {max(graph_insights['centrality']['degree'].items(), key=lambda x: x[1])[0]}
            - Key Intermediary: {max(graph_insights['centrality']['betweenness'].items(), key=lambda x: x[1])[0]}
            - Most Influential: {max(graph_insights['centrality']['eigenvector'].items(), key=lambda x: x[1])[0]}
            """

            # Prepare transaction context
            context = f"""
            Transaction Overview:
            - Volume: {len(filtered_transactions)} total transactions
            - Time Span: {filtered_transactions['timestamp'].min()} to {filtered_transactions['timestamp'].max()}
            - Amount Range: ${filtered_transactions['Amount'].min():.2f} to ${filtered_transactions['Amount'].max():.2f}
            - Risk Scores: {filtered_transactions['risk_score'].min():.2f} to {filtered_transactions['risk_score'].max():.2f}
            
            Network Size:
            - Active Accounts: {filtered_transactions['AccountID'].nunique()}
            - Active Merchants: {filtered_transactions['MerchantID'].nunique()}
            
            Key Metrics:
            - Average Transaction: ${filtered_transactions['Amount'].mean():.2f}
            - Median Transaction: ${filtered_transactions['Amount'].median():.2f}
            - Mean Risk Score: {filtered_transactions['risk_score'].mean():.2f}
            - High-Risk Count: {len(filtered_transactions[filtered_transactions['risk_score'] > 0.7])} transactions
            
            {pattern_info}
            """

            try:
                # Get response from chain with proper error handling
                response = self.chain({
                    "input": question,
                    "context": context,
                    "graph_info": graph_info
                })
                return response["output"], G
            except Exception as chain_error:
                st.error(f"Error getting response from LLM: {str(chain_error)}")
                return "I apologize, but I encountered an error while processing your question. Please try rephrasing or ask a different question.", G

        except Exception as e:
            error_msg = f"Error analyzing question: {str(e)}"
            st.error(error_msg)
            return error_msg, None

    def display_qa_interface(self, filtered_transactions):
        """Display the Q&A interface with enhanced visualization."""
        st.header("Transaction Analysis Q&A")
        
        # Example questions with categories
        st.subheader("Example Questions")
        question_categories = {
            "Pattern Analysis": [
                "What are the most common transaction patterns in the network?",
                "Are there any circular transaction patterns that might indicate money laundering?",
                "Which accounts show unusual transaction frequencies?"
            ],
            "Risk Assessment": [
                "What are the top 5 highest-risk transactions and their characteristics?",
                "Are there any merchants associated with multiple high-risk transactions?",
                "What is the relationship between transaction amounts and risk scores?"
            ],
            "Network Analysis": [
                "Who are the most central actors in the transaction network?",
                "Are there any isolated clusters of transactions?",
                "Which accounts have the most complex transaction patterns?"
            ]
        }
        
        for category, questions in question_categories.items():
            with st.expander(f"{category} Questions"):
                for q in questions:
                    st.markdown(f"- {q}")
        
        # User input with auto-complete
        user_question = st.text_input(
            "Ask a question about the transaction data:",
            placeholder="e.g., What are the highest-risk transactions?"
        )
        
        if user_question:
            with st.spinner("Analyzing transaction patterns..."):
                try:
                    # Get answer and updated graph
                    answer, G = self.process_qa(user_question, filtered_transactions)
                    
                    # Display answer in a structured way with proper formatting
                    st.markdown("### Analysis Results")
                    
                    # Split answer into sections if it contains numbered sections
                    sections = answer.split("\n\n")
                    for section in sections:
                        if section.strip():
                            # Add proper markdown formatting
                            if section.startswith(("1.", "2.", "3.", "4.")):
                                st.markdown(f"**{section.split('.')[0]}.** {section.split('.', 1)[1]}")
                            else:
                                st.markdown(section)
                            st.markdown("---")  # Add separator between sections
                    
                    # Show relevant visualizations based on the question
                    if G is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Relevant Data")
                            if "risk" in user_question.lower():
                                high_risk = filtered_transactions[filtered_transactions['risk_score'] > 0.7]
                                st.dataframe(high_risk.sort_values('risk_score', ascending=False))
                            elif "amount" in user_question.lower():
                                st.dataframe(filtered_transactions.sort_values('Amount', ascending=False))
                            elif "frequent" in user_question.lower():
                                tx_counts = filtered_transactions.groupby('AccountID').size().sort_values(ascending=False)
                                st.dataframe(tx_counts)
                        
                        with col2:
                            st.subheader("Network Metrics")
                            metrics = self.get_graph_insights(G)
                            col2_1, col2_2, col2_3 = st.columns(3)
                            with col2_1:
                                st.metric("Graph Density", f"{metrics['density']:.3f}")
                            with col2_2:
                                st.metric("Connected Components", metrics['components'])
                            with col2_3:
                                st.metric("Circular Patterns", metrics['cycles'])
                
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
                    st.info("Please try rephrasing your question or ask something else.")

    def load_data(self):
        """Load and preprocess transaction data."""
        if self.transactions is None:
            self.transactions = self.db_client.get_all_transactions()
            
            # Analyze all transactions
            for tx in self.transactions:
                analysis = self.fraud_analyzer.analyze_transaction(tx)
                tx['risk_score'] = analysis['risk_score']
                tx['risk_factors'] = analysis['risk_factors']
                tx['risk_details'] = analysis['risk_details']
                
            # Convert to DataFrame for easier filtering
            self.df = pd.DataFrame(self.transactions)
            
            # Convert timestamp to datetime
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Convert Amount to float
            self.df['Amount'] = self.df['Amount'].astype(float)

    def create_pyvis_network(self, min_risk=0.0, max_risk=1.0, min_amount=0, max_amount=float('inf'),
                           selected_accounts=None, selected_merchants=None, date_range=None):
        """Create an interactive Pyvis network visualization."""
        # Create NetworkX graph first
        G = nx.DiGraph()
        
        # Filter transactions
        mask = (
            (self.df['risk_score'] >= min_risk) &
            (self.df['risk_score'] <= max_risk) &
            (self.df['Amount'] >= min_amount) &
            (self.df['Amount'] <= max_amount)
        )
        
        if selected_accounts:
            mask &= self.df['AccountID'].isin(selected_accounts)
        
        if selected_merchants:
            mask &= self.df['MerchantID'].isin(selected_merchants)
            
        if date_range:
            mask &= (
                (self.df['timestamp'] >= date_range[0]) &
                (self.df['timestamp'] <= date_range[1])
            )
        
        filtered_transactions = self.df[mask]
        
        # Create Pyvis network
        net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Configure physics - adjusted for more spacing
        net.force_atlas_2based(
            gravity=-200,  # Increased negative gravity for more repulsion
            central_gravity=0.005,  # Reduced central gravity to allow nodes to spread out
            spring_length=300,  # Increased spring length for more distance between nodes
            spring_strength=0.02  # Reduced spring strength to allow more flexibility
        )
        
        # Configure other options
        net.set_options("""
{
    "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 3,
        "size": 30
    },
    "edges": {
        "color": {
            "inherit": true
        },
        "smooth": {
            "type": "continuous",
            "forceDirection": "none"
        }
    },
    "physics": {
        "barnesHut": {
            "gravitationalConstant": -150000,
            "springLength": 300,
            "springConstant": 0.02,
            "damping": 0.09,
            "avoidOverlap": 1
        },
        "minVelocity": 0.75,
        "solver": "barnesHut",
        "stabilization": {
            "enabled": true,
            "iterations": 2000
        }
    },
    "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
    }
}
""")
        
        # Add nodes and edges
        for _, tx in filtered_transactions.iterrows():
            # Add account node if not exists
            if not G.has_node(tx['AccountID']):
                net.add_node(
                    tx['AccountID'],
                    label=f"A: {tx['AccountID'][-4:]}",
                    title=f"Account: {tx['AccountID']}",
                    color='lightblue',
                    shape='circle',
                    size=35
                )
            
            # Add merchant node if not exists
            if not G.has_node(tx['MerchantID']):
                net.add_node(
                    tx['MerchantID'],
                    label=f"M: {tx['MerchantID'][-4:]}",
                    title=f"Merchant: {tx['MerchantID']}",
                    color='lightgreen',
                    shape='square',
                    size=35
                )
            
            # Add transaction node
            risk_color = self.get_risk_color(tx['risk_score'])
            node_size = min(25 + tx['Amount']/200, 45)
            
            net.add_node(
                tx['TransactionId'],
                label=f"T: {tx['TransactionId'][-4:]}",
                title=(f"Transaction: {tx['TransactionId']}<br>"
                       f"Amount: ${tx['Amount']:,.2f}<br>"
                       f"Risk Score: {tx['risk_score']:.2f}<br>"
                       f"Time: {tx['timestamp']}"),
                color=risk_color,
                shape='diamond',
                size=node_size
            )
            
            # Add edges
            net.add_edge(
                tx['AccountID'],
                tx['TransactionId'],
                value=tx['Amount'],
                title=f"${tx['Amount']:,.2f}"
            )
            net.add_edge(
                tx['TransactionId'],
                tx['MerchantID'],
                value=tx['Amount'],
                title=f"${tx['Amount']:,.2f}"
            )
        
        return net, filtered_transactions

    def get_risk_color(self, risk_score):
        """Convert risk score to color."""
        if risk_score >= 0.7:
            return '#ff4444'  # Red for high risk
        elif risk_score >= 0.4:
            return '#ffaa00'  # Orange for medium risk
        else:
            return '#00cc00'  # Green for low risk

    def analyze_transaction_patterns(self, filtered_transactions):
        """Analyze transaction patterns for suspicious activity."""
        patterns = {
            'circular_patterns': [],
            'unusual_amounts': [],
            'high_frequency': [],
            'central_nodes': {}
        }
        
        # Create NetworkX graph for analysis
        G = nx.DiGraph()
        for _, tx in filtered_transactions.iterrows():
            G.add_edge(tx['AccountID'], tx['MerchantID'], 
                      weight=tx['Amount'],
                      transaction_id=tx['TransactionId'],
                      risk_score=tx['risk_score'],
                      timestamp=tx['timestamp'])
        
        # Find circular patterns (cycles)
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) >= 3:  # Consider cycles of length 3 or more
                cycle_txs = []
                for i in range(len(cycle)):
                    node1, node2 = cycle[i], cycle[(i+1) % len(cycle)]
                    if G.has_edge(node1, node2):
                        edge_data = G.get_edge_data(node1, node2)
                        cycle_txs.append({
                            'transaction_id': edge_data['transaction_id'],
                            'amount': edge_data['weight'],
                            'risk_score': edge_data['risk_score']
                        })
                patterns['circular_patterns'].append({
                    'nodes': cycle,
                    'transactions': cycle_txs
                })
        
        # Detect unusual amounts
        amounts = filtered_transactions['Amount'].values
        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        threshold = mean_amount + 2 * std_amount
        unusual_txs = filtered_transactions[filtered_transactions['Amount'] > threshold]
        patterns['unusual_amounts'] = unusual_txs.to_dict('records')
        
        # Analyze high-frequency patterns
        time_diffs = filtered_transactions.groupby('AccountID')['timestamp'].diff()
        high_freq_mask = time_diffs < pd.Timedelta(minutes=5)
        high_freq_txs = filtered_transactions[high_freq_mask]
        patterns['high_frequency'] = high_freq_txs.to_dict('records')
        
        # Network centrality analysis
        patterns['central_nodes'] = {
            'degree': dict(nx.degree_centrality(G)),
            'betweenness': dict(nx.betweenness_centrality(G)),
            'eigenvector': dict(nx.eigenvector_centrality(G, max_iter=300))
        }
        
        return patterns, G

    def display_pattern_analysis(self, patterns):
        """Display the results of pattern analysis."""
        st.subheader("Suspicious Pattern Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "Circular Patterns", 
            "Unusual Amounts", 
            "High-Frequency Trading",
            "Network Centrality"
        ])
        
        with tab1:
            st.write(f"Found {len(patterns['circular_patterns'])} circular transaction patterns")
            for i, pattern in enumerate(patterns['circular_patterns']):
                with st.expander(f"Circular Pattern {i+1} - {len(pattern['nodes'])} nodes"):
                    st.write("Path:", " → ".join(pattern['nodes']))
                    st.dataframe(pd.DataFrame(pattern['transactions']))
        
        with tab2:
            st.write(f"Found {len(patterns['unusual_amounts'])} unusual transaction amounts")
            if patterns['unusual_amounts']:
                df = pd.DataFrame(patterns['unusual_amounts'])
                st.dataframe(df.sort_values('Amount', ascending=False))
        
        with tab3:
            st.write(f"Found {len(patterns['high_frequency'])} high-frequency transactions")
            if patterns['high_frequency']:
                df = pd.DataFrame(patterns['high_frequency'])
                st.dataframe(df.sort_values('timestamp'))
        
        with tab4:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Top Degree Centrality")
                central_degree = sorted(patterns['central_nodes']['degree'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]
                st.dataframe(pd.DataFrame(central_degree, 
                                        columns=['Node', 'Centrality']))
            
            with col2:
                st.write("Top Betweenness Centrality")
                central_between = sorted(patterns['central_nodes']['betweenness'].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]
                st.dataframe(pd.DataFrame(central_between, 
                                        columns=['Node', 'Centrality']))
            
            with col3:
                st.write("Top Eigenvector Centrality")
                central_eigen = sorted(patterns['central_nodes']['eigenvector'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                st.dataframe(pd.DataFrame(central_eigen, 
                                        columns=['Node', 'Centrality']))

    def run(self):
        """Run the Streamlit app."""
        st.title("Transaction Network Analysis")
        
        # Create tabs for different functionalities
        tab1, tab2 = st.tabs(["Network Analysis", "Q&A Assistant"])
        
        with tab1:
            # Load data
            self.load_data()
            
            # Sidebar filters
            st.sidebar.header("Filters")
            
            # Risk score filter
            st.sidebar.subheader("Risk Score Range")
            min_risk, max_risk = st.sidebar.slider(
                "Select risk score range:",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.1
            )
            
            # Amount filter
            st.sidebar.subheader("Amount Range")
            max_amount = float(self.df['Amount'].max())
            min_amount, amount_threshold = st.sidebar.slider(
                "Select amount range ($):",
                min_value=0.0,
                max_value=max_amount,
                value=(0.0, max_amount),
                step=100.0
            )
            
            # Date range filter
            st.sidebar.subheader("Date Range")
            min_date = self.df['timestamp'].min()
            max_date = self.df['timestamp'].max()
            start_date = st.sidebar.date_input("Start date", min_date)
            end_date = st.sidebar.date_input("End date", max_date)
            
            # Account filter
            st.sidebar.subheader("Filter by Accounts")
            selected_accounts = st.sidebar.multiselect(
                "Select specific accounts:",
                options=sorted(self.df['AccountID'].unique())
            )
            
            # Merchant filter
            st.sidebar.subheader("Filter by Merchants")
            selected_merchants = st.sidebar.multiselect(
                "Select specific merchants:",
                options=sorted(self.df['MerchantID'].unique())
            )
            
            # Create network with filters
            net, filtered_transactions = self.create_pyvis_network(
                min_risk=min_risk,
                max_risk=max_risk,
                min_amount=min_amount,
                max_amount=amount_threshold,
                selected_accounts=selected_accounts if selected_accounts else None,
                selected_merchants=selected_merchants if selected_merchants else None,
                date_range=[pd.Timestamp(start_date), pd.Timestamp(end_date)]
            )
            
            # Save and display network
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                net.save_graph(tmp_file.name)
                with open(tmp_file.name, 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=800, width=None)
            
            # Analyze and display patterns
            patterns, G = self.analyze_transaction_patterns(filtered_transactions)
            self.display_pattern_analysis(patterns)
            
            # Statistics
            st.subheader("Network Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transactions", len(filtered_transactions))
            
            with col2:
                st.metric("Unique Accounts", filtered_transactions['AccountID'].nunique())
            
            with col3:
                st.metric("Unique Merchants", filtered_transactions['MerchantID'].nunique())
            
            # Transaction details
            st.subheader("Transaction Details")
            filtered_df = filtered_transactions.sort_values('risk_score', ascending=False)
            
            # Display as table
            st.dataframe(
                filtered_df[[
                    'TransactionId', 'timestamp', 'Amount', 'AccountID',
                    'MerchantID', 'risk_score', 'risk_factors'
                ]].style.format({
                    'Amount': '${:,.2f}',
                    'risk_score': '{:.2f}'
                })
            )
        
        with tab2:
            self.display_qa_interface(filtered_transactions)

if __name__ == "__main__":
    app = InteractiveGraphApp()
    app.run() 