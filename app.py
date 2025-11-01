import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page configuration with Zerodha-like color scheme
st.set_page_config(
    page_title="Portfolio Returns and Risk Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Zerodha-inspired color scheme
primary_color = "#387ed1"  # Zerodha blue
secondary_color = "#2ecc71"  # Green for positive
accent_color = "#e74c3c"   # Red for negative
background_color = "#f8f9fa"
card_color = "#ffffff"
text_color = "#2c3e50"

# Custom CSS with Zerodha-like styling
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        color: {primary_color};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, {primary_color}, #2c3e50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .section-header {{
        font-size: 1.8rem;
        color: {primary_color};
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid {primary_color};
        padding-bottom: 0.5rem;
        font-weight: 600;
    }}
    .metric-card {{
        background-color: {card_color};
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 5px solid {primary_color};
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }}
    .risk-conservative {{
        color: #27ae60;
        font-weight: bold;
        background-color: #d5f4e6;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #27ae60;
    }}
    .risk-moderate {{
        color: #f39c12;
        font-weight: bold;
        background-color: #fef5e7;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #f39c12;
    }}
    .risk-aggressive {{
        color: #e67e22;
        font-weight: bold;
        background-color: #fdebd0;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #e67e22;
    }}
    .risk-high {{
        color: #e74c3c;
        font-weight: bold;
        background-color: #fadbd8;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #e74c3c;
    }}
    .recommendation-box {{
        background: linear-gradient(135deg, {primary_color}, #3498db);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }}
    .stock-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }}
    .stock-card:hover {{
        transform: scale(1.02);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }}
    .positive-return {{
        color: {secondary_color};
        font-weight: bold;
    }}
    .negative-return {{
        color: {accent_color};
        font-weight: bold;
    }}
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, {primary_color}, #2c3e50);
    }}
    .stButton>button {{
        background: linear-gradient(135deg, {primary_color}, #3498db);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, #3498db, {primary_color});
        transform: translateY(-2px);
    }}
</style>
""", unsafe_allow_html=True)

class StockPortfolioApp:
    def __init__(self):
        self.models_loaded = False
        self.portfolios = {}
        self.portfolio_performance = {}
        self.nifty50_stocks = self.get_nifty50_stocks()
        self.load_models()
        
    def get_nifty50_stocks(self):
        """Get actual NIFTY 50 stock list"""
        nifty50_stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'HDFC', 'ICICIBANK',
            'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'ASIANPAINT', 'DMART', 'BAJFINANCE',
            'MARUTI', 'TITAN', 'SUNPHARMA', 'TATAMOTORS', 'ULTRACEMCO', 'NESTLEIND',
            'ONGC', 'LT', 'HCLTECH', 'BAJAJFINSV', 'WIPRO', 'ADANIPORTS', 'POWERGRID',
            'NTPC', 'M&M', 'AXISBANK', 'TECHM', 'TATASTEEL', 'JSWSTEEL', 'HDFCLIFE',
            'DRREDDY', 'BRITANNIA', 'GRASIM', 'CIPLA', 'COALINDIA', 'IOC', 'SHREECEM',
            'HINDALCO', 'INDUSINDBK', 'DIVISLAB', 'SBILIFE', 'UPL', 'BAJAJ-AUTO',
            'HEROMOTOCO', 'EICHERMOT', 'APOLLOHOSP'
        ]
        return nifty50_stocks
    
    def load_models(self):
        """Load trained models and preprocessing objects"""
        try:
            # Risk classification models
            self.log_reg_risk = joblib.load('logistic_regression_risk_model.pkl')
            self.rf_risk = joblib.load('random_forest_risk_model.pkl')
            
            # Return prediction models
            self.lr_return = joblib.load('linear_regression_return_model.pkl')
            self.rf_return = joblib.load('random_forest_return_model.pkl')
            
            # Preprocessing objects
            self.scaler_risk = joblib.load('scaler_risk.pkl')
            self.scaler_return = joblib.load('scaler_return.pkl')
            self.le_risk = joblib.load('label_encoder_risk.pkl')
            self.risk_feature_names = joblib.load('risk_feature_names.pkl')
            self.return_feature_names = joblib.load('return_feature_names.pkl')
            
            # Portfolio information
            try:
                self.portfolio_info = joblib.load('portfolio_info.pkl')
                self.portfolios = self.portfolio_info.get('portfolios', {})
                self.portfolio_performance = self.portfolio_info.get('portfolio_performance', {})
            except FileNotFoundError:
                st.warning("Portfolio info file not found. Creating comprehensive NIFTY 50 portfolios...")
                self.create_nifty50_portfolios()
            
            self.models_loaded = True
            st.success("All models loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.info("Creating comprehensive NIFTY 50 portfolios for demonstration...")
            self.create_nifty50_portfolios()
    
    def create_nifty50_portfolios(self):
        """Create comprehensive portfolios with all NIFTY 50 stocks"""
        np.random.seed(42)
        
        # Create realistic stock data for NIFTY 50
        stock_data = []
        for i, company in enumerate(self.nifty50_stocks):
            beta = np.random.uniform(0.5, 1.8)
            volatility = np.random.uniform(0.15, 0.45)
            
            # Calculate expected return using CAPM
            risk_free_rate = 0.06
            market_return = 0.12
            expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
            
            # Add some randomness to actual returns
            actual_return = expected_return + np.random.normal(0, 0.03)
            
            stock_data.append({
                'Company': company,
                'Expected Return': expected_return,
                'Actual Return': actual_return,
                'Volatility': volatility,
                'Beta': beta,
                'PE Ratio': np.random.uniform(15, 50),
                'Market Cap (Cr)': np.random.uniform(50000, 500000),
                'Current Price': np.random.uniform(100, 5000)
            })
        
        stocks_df = pd.DataFrame(stock_data)
        
        # Enhanced Risk Classification
        def get_risk_category(beta, volatility):
            risk_score = beta * 0.6 + volatility * 0.4
            if risk_score < 0.7:
                return "Conservative"
            elif risk_score < 1.0:
                return "Moderate"
            elif risk_score < 1.3:
                return "Aggressive"
            else:
                return "High-Risk"
        
        stocks_df['Risk Category'] = stocks_df.apply(
            lambda x: get_risk_category(x['Beta'], x['Volatility']), axis=1
        )
        
        # Create portfolios
        self.portfolios = {}
        for risk_category in ['Conservative', 'Moderate', 'Aggressive', 'High-Risk']:
            category_stocks = stocks_df[stocks_df['Risk Category'] == risk_category]
            if len(category_stocks) > 0:
                # Create smart allocation (higher allocation to better risk-adjusted returns)
                sharpe_ratios = (category_stocks['Expected Return'] - 0.06) / category_stocks['Volatility']
                allocations = sharpe_ratios / sharpe_ratios.sum()
                
                self.portfolios[risk_category] = {
                    'stocks': category_stocks,
                    'allocation': allocations.values,
                    'expected_return': category_stocks['Expected Return'].mean(),
                    'risk': category_stocks['Volatility'].mean(),
                    'sharpe_ratio': (category_stocks['Expected Return'].mean() - 0.06) / category_stocks['Volatility'].mean()
                }
    
    def predict_stock_risk(self, features, model_type='rf'):
        """Predict risk category for a stock"""
        try:
            if not self.models_loaded:
                # Return based on beta value for demo
                beta = features[3] if len(features) > 3 else 1.0
                if beta < 0.8: return "Conservative"
                elif beta < 1.2: return "Moderate"
                elif beta < 1.5: return "Aggressive"
                else: return "High-Risk"
            
            if model_type == 'rf':
                model = self.rf_risk
            else:
                model = self.log_reg_risk
            
            # Ensure features match expected length
            if len(features) != len(self.risk_feature_names):
                features = features[:len(self.risk_feature_names)]
                if len(features) < len(self.risk_feature_names):
                    features.extend([0.0] * (len(self.risk_feature_names) - len(features)))
            
            # Scale features
            features_scaled = self.scaler_risk.transform([features])
            
            # Predict
            prediction_encoded = model.predict(features_scaled)[0]
            prediction = self.le_risk.inverse_transform([prediction_encoded])[0]
            
            return prediction
        except Exception:
            # Fallback to beta-based classification
            beta = features[3] if len(features) > 3 else 1.0
            if beta < 0.8: return "Conservative"
            elif beta < 1.2: return "Moderate"
            elif beta < 1.5: return "Aggressive"
            else: return "High-Risk"
    
    def predict_stock_return(self, features, model_type='rf'):
        """Predict expected return for a stock"""
        try:
            if not self.models_loaded:
                # CAPM-based return for demo
                beta = features[0] if len(features) > 0 else 1.0
                return 0.06 + beta * (0.12 - 0.06)
            
            if model_type == 'rf':
                model = self.rf_return
            else:
                model = self.lr_return
            
            # Ensure features match expected length
            if len(features) != len(self.return_feature_names):
                features = features[:len(self.return_feature_names)]
                if len(features) < len(self.return_feature_names):
                    features.extend([0.0] * (len(self.return_feature_names) - len(features)))
            
            # Scale features
            features_scaled = self.scaler_return.transform([features])
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            
            return prediction
        except Exception:
            # CAPM-based return for demo
            beta = features[0] if len(features) > 0 else 1.0
            return 0.06 + beta * (0.12 - 0.06)
    
    def get_risk_class_color(self, risk_category):
        """Get CSS class for risk category"""
        risk_colors = {
            'Conservative': 'risk-conservative',
            'Moderate': 'risk-moderate',
            'Aggressive': 'risk-aggressive',
            'High-Risk': 'risk-high'
        }
        return risk_colors.get(risk_category, 'risk-moderate')

def main():
    # Initialize app
    app = StockPortfolioApp()
    
    # Main header
    st.markdown('<div class="main-header">📊 Portfolio Returns and Risk Analyser</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("### 🔍 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📈 Dashboard", "🔍 Stock Analysis", "💼 Portfolio Builder", "🎯 Investment Recommendations"]
    )
    
    # Dashboard
    if app_mode == "📈 Dashboard":
        show_dashboard(app)
    
    # Stock Analysis
    elif app_mode == "🔍 Stock Analysis":
        show_stock_analysis(app)
    
    # Portfolio Builder
    elif app_mode == "💼 Portfolio Builder":
        show_portfolio_builder(app)
    
    # Investment Recommendations
    elif app_mode == "🎯 Investment Recommendations":
        show_investment_recommendations(app)

def show_dashboard(app):
    st.markdown("## 📈 Portfolio Dashboard")
    
    # Portfolio Overview Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stocks = sum(len(portfolio['stocks']) for portfolio in app.portfolios.values())
        st.metric("Total NIFTY 50 Stocks", total_stocks)
    
    with col2:
        total_portfolios = len(app.portfolios)
        st.metric("Risk Portfolios", total_portfolios)
    
    with col3:
        returns = [portfolio['expected_return'] for portfolio in app.portfolios.values()]
        avg_return = np.mean(returns) if returns else 0
        st.metric("Avg Expected Return", f"{avg_return:.2%}")
    
    with col4:
        risks = [portfolio['risk'] for portfolio in app.portfolios.values()]
        avg_risk = np.mean(risks) if risks else 0
        st.metric("Avg Risk", f"{avg_risk:.2%}")
    
    # Portfolio Performance
    st.markdown("### 💼 Portfolio Performance Overview")
    
    for portfolio_name, portfolio_data in app.portfolios.items():
        if len(portfolio_data['stocks']) > 0:
            st.markdown(f"#### {portfolio_name} Portfolio")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Stocks", len(portfolio_data['stocks']))
            
            with col2:
                st.metric("Expected Return", f"{portfolio_data['expected_return']:.2%}")
            
            with col3:
                st.metric("Risk", f"{portfolio_data['risk']:.2%}")
            
            with col4:
                sharpe_ratio = portfolio_data.get('sharpe_ratio', 
                    (portfolio_data['expected_return'] - 0.06) / portfolio_data['risk'] if portfolio_data['risk'] > 0 else 0)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            # Show top stocks in this portfolio
            st.markdown("**Top Stocks:**")
            stocks_display = portfolio_data['stocks'][['Company', 'Expected Return', 'Risk Category']].head(8)
            for _, stock in stocks_display.iterrows():
                st.markdown(f"• **{stock['Company']}** - {stock['Expected Return']:.2%} expected return")
    
    # Risk-Return Visualization
    st.markdown("### 📊 Risk-Return Analysis")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {'Conservative': '#27ae60', 'Moderate': '#f39c12', 'Aggressive': '#e67e22', 'High-Risk': '#e74c3c'}
    
    # Plot all NIFTY 50 stocks
    all_stocks = []
    for portfolio_name, portfolio_data in app.portfolios.items():
        stocks_df = portfolio_data['stocks']
        if len(stocks_df) > 0:
            ax.scatter(stocks_df['Volatility'], stocks_df['Expected Return'], 
                      c=colors[portfolio_name], s=100, label=portfolio_name, alpha=0.7)
            all_stocks.append(stocks_df)
    
    if all_stocks:
        combined_df = pd.concat(all_stocks, ignore_index=True)
        
        # Add labels for major stocks
        major_stocks = combined_df.nlargest(10, 'Market Cap (Cr)') if 'Market Cap (Cr)' in combined_df.columns else combined_df.head(10)
        for _, stock in major_stocks.iterrows():
            ax.annotate(stock['Company'], 
                       (stock['Volatility'], stock['Expected Return']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('NIFTY 50 Stocks: Risk-Return Profile', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)

def show_stock_analysis(app):
    st.markdown("## 🔍 Stock Analysis")
    
    # Stock selection
    selected_stock = st.selectbox("Select NIFTY 50 Stock:", app.nifty50_stocks)
    
    if st.button("Analyze Stock", type="primary"):
        with st.spinner("Fetching stock data..."):
            try:
                # Add .NS suffix for NSE stocks
                ticker = f'{selected_stock}.NS'
                stock = yf.Ticker(ticker)
                
                # Get live data
                hist = stock.history(period='1d')
                if hist.empty:
                    st.error("No data available for this stock.")
                    return
                
                live_price = hist['Close'].iloc[-1]
                
                # Get historical data for volatility calculation
                hist_1y = stock.history(period='1y')
                if len(hist_1y) > 0:
                    returns = hist_1y['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                else:
                    volatility = 0.25
                
                # Get fundamentals - with better error handling
                info = stock.info
                beta = info.get('beta', 1.0)
                pe_ratio = info.get('trailingPE', 20.0)
                market_cap = info.get('marketCap', 100000) / 10000000
                
                # Better dividend yield calculation
                try:
                    dividend_yield = info.get('dividendYield', 0.0)
                    if dividend_yield is None:
                        dividend_yield = 0.0
                except:
                    dividend_yield = 0.0
                
                # Display stock information
                st.markdown("### Stock Information")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"₹{live_price:.2f}")
                    st.metric("Beta", f"{beta:.2f}")
                
                with col2:
                    st.metric("PE Ratio", f"{pe_ratio:.2f}")
                    st.metric("Volatility", f"{volatility:.2%}")
                
                with col3:
                    st.metric("Market Cap (Cr.)", f"₹{market_cap:,.0f}")
                    st.metric("Dividend Yield", f"{dividend_yield:.2%}")
                
                with col4:
                    # Calculate additional metrics
                    risk_free_rate = 0.06
                    market_return = 0.12
                    expected_return_capm = risk_free_rate + beta * (market_return - risk_free_rate)
                    st.metric("CAPM Expected Return", f"{expected_return_capm:.2%}")
                
                # Prepare features for ML prediction
                risk_features = [live_price, live_price*0.99, 1000000, beta, volatility, 
                               pe_ratio, dividend_yield, 60.0, 500.0, 1, market_cap]
                
                return_features = [beta, volatility, pe_ratio, dividend_yield, market_cap]
                
                # Get predictions
                risk_prediction = app.predict_stock_risk(risk_features, 'rf')
                return_prediction = app.predict_stock_return(return_features, 'rf')
                
                # Display Analysis Results
                st.markdown("### Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f'<div class="{app.get_risk_class_color(risk_prediction)}">', unsafe_allow_html=True)
                    st.subheader("Risk Category")
                    st.write(f"**{risk_prediction}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Expected Return")
                    st.metric("ML Prediction", f"{return_prediction:.2%}")
                    st.metric("CAPM Model", f"{expected_return_capm:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Investment Recommendation
                st.markdown("### 💡 Investment Recommendation")
                
                recommendation_text = {
                    "Conservative": """
                    **🛡️ Conservative Investment - Suitable for risk-averse investors**
                    - Focus on capital preservation
                    - Stable, predictable returns
                    - Lower volatility
                    - Good for long-term wealth building
                    """,
                    "Moderate": """
                    **⚖️ Moderate Investment - Balanced risk-return profile**
                    - Mix of growth and stability
                    - Moderate volatility
                    - Suitable for most investors
                    - Good for medium to long-term goals
                    """,
                    "Aggressive": """
                    **🚀 Aggressive Investment - Growth-focused**
                    - Higher growth potential
                    - Increased volatility
                    - Suitable for growth-oriented investors
                    - Requires active monitoring
                    """,
                    "High-Risk": """
                    **🔥 High-Risk Investment - Speculative**
                    - Highest growth potential
                    - Significant volatility
                    - Suitable for experienced investors
                    - Requires careful risk management
                    """
                }
                
                st.markdown(f'<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown(recommendation_text.get(risk_prediction, recommendation_text["Moderate"]))
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error analyzing stock: {str(e)}")

def show_portfolio_builder(app):
    st.markdown("## 💼 Smart Portfolio Builder")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_tolerance = st.select_slider(
            "Select Your Risk Tolerance:",
            options=["Very Low", "Low", "Medium", "High", "Very High"],
            value="Medium"
        )
    
    with col2:
        time_duration = st.select_slider(
            "Investment Time Duration:",
            options=["Short-term (<1 year)", "Medium-term (1-3 years)", "Long-term (3-5 years)", "Very Long-term (>5 years)"],
            value="Long-term (3-5 years)"
        )
    
    investment_amount = st.number_input("Investment Amount (₹):", min_value=1000, value=100000, step=1000)
    
    # Map risk tolerance to portfolio
    risk_mapping = {
        'Very Low': 'Conservative',
        'Low': 'Conservative',
        'Medium': 'Moderate',
        'High': 'Aggressive',
        'Very High': 'High-Risk'
    }
    
    selected_portfolio = risk_mapping[risk_tolerance]
    
    if st.button("Build Portfolio", type="primary"):
        if selected_portfolio in app.portfolios and len(app.portfolios[selected_portfolio]['stocks']) > 0:
            portfolio_data = app.portfolios[selected_portfolio]
            
            st.markdown(f"### Recommended: {selected_portfolio} Portfolio")
            
            # Portfolio statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Return", f"{portfolio_data['expected_return']:.2%}")
            
            with col2:
                st.metric("Risk (Volatility)", f"{portfolio_data['risk']:.2%}")
            
            with col3:
                sharpe_ratio = portfolio_data.get('sharpe_ratio', 
                    (portfolio_data['expected_return'] - 0.06) / portfolio_data['risk'] if portfolio_data['risk'] > 0 else 0)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            with col4:
                st.metric("Time Horizon", time_duration.split('(')[1].replace(')', ''))
            
            # Stock allocation table
            st.markdown("### 📋 Portfolio Allocation")
            
            stocks_df = portfolio_data['stocks'].copy()
            if 'allocation' in portfolio_data and len(portfolio_data['allocation']) > 0:
                # Ensure allocation matches number of stocks
                n_stocks = len(stocks_df)
                if len(portfolio_data['allocation']) >= n_stocks:
                    stocks_df['Allocation %'] = portfolio_data['allocation'][:n_stocks] * 100
                else:
                    stocks_df['Allocation %'] = 100 / n_stocks
                
                stocks_df['Investment (₹)'] = stocks_df['Allocation %'] * investment_amount / 100
            
            # Display stock details
            display_columns = ['Company', 'Expected Return', 'Volatility', 'Beta']
            if 'Allocation %' in stocks_df.columns:
                display_columns.extend(['Allocation %', 'Investment (₹)'])
            
            # Format the dataframe for display
            display_df = stocks_df[display_columns].copy()
            display_df['Expected Return'] = display_df['Expected Return'].apply(lambda x: f"{x:.2%}")
            display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.2%}")
            if 'Investment (₹)' in display_df.columns:
                display_df['Investment (₹)'] = display_df['Investment (₹)'].apply(lambda x: f"₹{x:,.0f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Top allocations pie chart
                if 'Allocation %' in stocks_df.columns:
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    top_stocks = stocks_df.nlargest(8, 'Allocation %')
                    ax1.pie(top_stocks['Allocation %'], labels=top_stocks['Company'], 
                           autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Top Stock Allocations')
                    st.pyplot(fig1)
            
            with col2:
                # Risk-return scatter
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.scatter(stocks_df['Volatility'], stocks_df['Expected Return'], 
                           alpha=0.6, s=100)
                ax2.set_xlabel('Volatility (Risk)')
                ax2.set_ylabel('Expected Return')
                ax2.set_title('Portfolio Stocks: Risk-Return Profile')
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
        
        else:
            st.warning(f"No stocks available in the {selected_portfolio} portfolio.")

def show_investment_recommendations(app):
    st.markdown("## 🎯 Personalized Investment Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_tolerance = st.selectbox(
            "What is your risk tolerance?",
            ["Very Low", "Low", "Medium", "High", "Very High"],
            index=2
        )
    
    with col2:
        investment_horizon = st.selectbox(
            "Investment Horizon:",
            ["Short-term (1-2 years)", "Medium-term (3-5 years)", "Long-term (5+ years)"]
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        age = st.slider("Your Age:", min_value=18, max_value=80, value=35)
    
    with col4:
        investment_amount = st.number_input("Investment Amount (₹):", 
                                          min_value=1000, value=100000, step=1000)
    
    investment_goal = st.selectbox(
        "Primary Investment Goal:",
        ["Wealth Preservation", "Regular Income", "Wealth Growth", "Aggressive Growth"]
    )
    
    if st.button("Generate Personalized Recommendation", type="primary"):
        # Map risk tolerance to portfolio
        risk_mapping = {
            'Very Low': 'Conservative',
            'Low': 'Conservative',
            'Medium': 'Moderate',
            'High': 'Aggressive',
            'Very High': 'High-Risk'
        }
        
        recommended_portfolio = risk_mapping.get(risk_tolerance, 'Moderate')
        
        if recommended_portfolio not in app.portfolios:
            st.error(f"Recommended portfolio '{recommended_portfolio}' not available.")
            return
        
        portfolio_data = app.portfolios[recommended_portfolio]
        
        # Age-based advice
        if age < 30:
            age_advice = "Young investors can afford to take more risks for long-term growth."
            suggested_adjustment = "Consider adding some aggressive stocks to your portfolio."
        elif age < 50:
            age_advice = "Mid-career investors should balance growth with stability."
            suggested_adjustment = "Maintain a diversified portfolio across risk categories."
        else:
            age_advice = "Pre-retirement investors should focus on capital preservation."
            suggested_adjustment = "Consider shifting towards more conservative investments."
        
        # Goal-based advice
        goal_advice = {
            "Wealth Preservation": "Focus on stable, dividend-paying stocks with low volatility.",
            "Regular Income": "Prioritize high-dividend yield stocks and consistent performers.",
            "Wealth Growth": "Balance between growth stocks and stable blue-chip companies.",
            "Aggressive Growth": "Focus on high-growth potential stocks across sectors."
        }
        
        # Display recommendation
        st.markdown("### 🎯 Your Personalized Investment Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Recommended Portfolio", recommended_portfolio)
            st.metric("Expected Return", f"{portfolio_data['expected_return']:.2%}")
            st.metric("Risk Level", f"{portfolio_data['risk']:.2%}")
        
        with col2:
            st.metric("Investment Amount", f"₹{investment_amount:,}")
            st.metric("Time Horizon", investment_horizon)
            st.metric("Number of Stocks", len(portfolio_data['stocks']))
        
        # Detailed recommendation
        st.markdown("### 📋 Portfolio Composition")
        
        # Show top recommended stocks
        stocks_df = portfolio_data['stocks']
        if 'allocation' in portfolio_data and len(portfolio_data['allocation']) > 0:
            n_stocks = len(stocks_df)
            if len(portfolio_data['allocation']) >= n_stocks:
                stocks_df['Allocation %'] = portfolio_data['allocation'][:n_stocks] * 100
            else:
                stocks_df['Allocation %'] = 100 / n_stocks
            
            stocks_df['Investment (₹)'] = stocks_df['Allocation %'] * investment_amount / 100
        
        # Display top 10 stocks
        top_stocks = stocks_df.nlargest(10, 'Allocation %' if 'Allocation %' in stocks_df.columns else 'Expected Return')
        
        for _, stock in top_stocks.iterrows():
            investment_amt = stock['Investment (₹)'] if 'Investment (₹)' in stock else investment_amount / 10
            st.markdown(f"""
            <div class="stock-card">
                <strong>{stock['Company']}</strong><br>
                Expected Return: {stock['Expected Return']:.2%} | 
                Risk: {stock['Volatility']:.2%} |
                Investment: ₹{investment_amt:,.0f}
            </div>
            """, unsafe_allow_html=True)
        
        # Strategic advice
        st.markdown("### 💡 Strategic Advice")
        st.markdown(f"""
        <div class="recommendation-box">
            <strong>Based on your profile:</strong><br><br>
            • <strong>Age Factor:</strong> {age_advice}<br>
            • <strong>Risk Adjustment:</strong> {suggested_adjustment}<br>
            • <strong>Goal Strategy:</strong> {goal_advice[investment_goal]}<br><br>
            
            <strong>Recommended Action:</strong> Invest ₹{investment_amount:,} in the {recommended_portfolio} portfolio 
            for {investment_horizon.lower()} to achieve your {investment_goal.lower()} goals.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
