# Transaction Network Analysis Application

An interactive web application for analyzing and visualizing transaction networks to detect potential fraud patterns and suspicious activities.

## Features

### 1. Interactive Network Visualization
- Dynamic force-directed graph layout
- Color-coded nodes representing:
  - Light blue circles: Account nodes
  - Light green squares: Merchant nodes
  - Colored diamonds: Transactions (green to red based on risk score)
- Node size varies with transaction amount
- Interactive controls:
  - Zoom in/out with mouse wheel
  - Pan by clicking and dragging
  - Hover for detailed information
  - Node selection and highlighting

### 2. Pattern Detection
- **Circular Transaction Patterns**
  - Identifies cycles of 3+ nodes
  - Helps detect potential money laundering schemes
  - Shows complete transaction paths and details

- **Unusual Amount Detection**
  - Identifies transactions above 2 standard deviations from mean
  - Highlights potentially suspicious large transfers
  - Sorted by amount for easy analysis

- **High-Frequency Trading Analysis**
  - Detects rapid transactions (within 5 minutes)
  - Groups by account to identify unusual trading patterns
  - Temporal analysis of transaction sequences

- **Network Centrality Analysis**
  - Degree Centrality: Most connected nodes
  - Betweenness Centrality: Key intermediary nodes
  - Eigenvector Centrality: Nodes connected to important nodes

### 3. Filtering Capabilities
- Risk Score Range
- Transaction Amount Range
- Date Range Selection
- Account/Merchant Filtering
- Real-time graph updates

### 4. Transaction Statistics
- Total Transaction Count
- Unique Accounts
- Unique Merchants
- Detailed Transaction Table

### 5. AI-Powered Q&A Assistant
- Natural language queries about transaction data
- Context-aware responses using LLM
- Example questions for quick start
- Automatic display of relevant data tables
- Real-time analysis of:
  - Risk patterns
  - Transaction amounts
  - Account activities
  - Merchant behaviors
  - Temporal patterns

## Setup

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Database Configuration**
   - Ensure ArangoDB is installed and running
   - Database will be automatically initialized on first run

3. **OpenAI API Configuration**
   ```bash
   # Set your GOOGLE API key
   export GOOGLE_API_KEY='your-api-key-here'
   ```

4. **Running the Application**
   ```bash
   # Set Python path and run Streamlit app
   PYTHONPATH=src streamlit run src/app.py --server.address 0.0.0.0
   ```

## Usage Guide

1. **Filtering Data**
   - Use sidebar controls to filter:
     - Risk scores (0.0 to 1.0)
     - Transaction amounts
     - Date ranges
     - Specific accounts or merchants

2. **Analyzing Patterns**
   - Navigate through the tabs below the network visualization:
     - "Circular Patterns" for potential money laundering
     - "Unusual Amounts" for suspicious transfers
     - "High-Frequency Trading" for rapid transaction sequences
     - "Network Centrality" for key nodes analysis

3. **Investigating Transactions**
   - Click nodes for detailed information
   - Use the transaction details table for sorting and filtering
   - Examine risk factors and scores

4. **Using the Q&A Assistant**
   - Navigate to the "Q&A Assistant" tab
   - View example questions for guidance
   - Type your question in natural language
   - Get AI-powered analysis and insights
   - View supporting data tables automatically
   - Ask follow-up questions for deeper analysis

## Technical Details

### Dependencies
- Streamlit: Web application framework
- NetworkX: Network analysis
- Pyvis: Interactive network visualization
- Pandas: Data manipulation
- NumPy: Numerical computations
- ArangoDB: Database backend
- LangChain: LLM integration
- Google Gemini: AI model provider

### Architecture
- `InteractiveGraphApp`: Main application class
- `FraudAnalyzer`: Transaction risk analysis
- `ArangoDBClient`: Database interactions
- LLM Integration: AI-powered Q&A system
- Modular design for easy extension

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.