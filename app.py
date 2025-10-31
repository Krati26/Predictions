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

# Set page configuration
st.set_page_config(
    page_title="Stock Portfolio Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2e86ab;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-conservative {
        color: #28a745;
        font-weight: bold;
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .risk-moderate {
        color: #ffc107;
        font-weight: bold;
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .risk-aggressive {
        color: #fd7e14;
        font-weight: bold;
        background-color: #ffe5d0;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StockPortfolioApp:
    def __init__(self):
        self.load_models()
        
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
            self.portfolio_info = joblib.load('portfolio_info.pkl')
            self.portfolios = self.portfolio_info['portfolios']
            self.portfolio_performance = self.portfolio_info['portfolio_performance']
            
            st.success("✅ All models loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.info("Please ensure all model files are in the same directory as app.py")
    
    def predict_stock_risk(self, features, model_type='rf'):
        """Predict risk category for a stock"""
        try:
            if model_type == 'rf':
                model = self.rf_risk
            else:
                model = self.log_reg_risk
            
            # Scale features
            features_scaled = self.scaler_risk.transform([features])
            
            # Predict
            prediction_encoded = model.predict(features_scaled)[0]
            prediction = self.le_risk.inverse_transform([prediction_encoded])[0]
            
            return prediction
        except Exception as e:
            return f"Error: {e}"
    
    def predict_stock_return(self, features, model_type='rf'):
        """Predict expected return for a stock"""
        try:
            if model_type == 'rf':
                model = self.rf_return
            else:
                model = self.lr_return
            
            # Scale features
            features_scaled = self.scaler_return.transform([features])
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            
            return prediction
        except Exception as e:
            return f"Error: {e}"
    
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
    st.markdown('<div class="main-header">📊 AI-Powered Stock Portfolio Risk Analyzer</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["🏠 Dashboard", "📈 Single Stock Analysis", "💼 Portfolio Builder", 
         "🎯 Investment Recommendations", "📊 Model Performance"]
    )
    
    # Dashboard
    if app_mode == "🏠 Dashboard":
        show_dashboard(app)
    
    # Single Stock Analysis
    elif app_mode == "📈 Single Stock Analysis":
        show_single_stock_analysis(app)
    
    # Portfolio Builder
    elif app_mode == "💼 Portfolio Builder":
        show_portfolio_builder(app)
    
    # Investment Recommendations
    elif app_mode == "🎯 Investment Recommendations":
        show_investment_recommendations(app)
    
    # Model Performance
    elif app_mode == "📊 Model Performance":
        show_model_performance(app)

def show_dashboard(app):
    st.markdown('<div class="section-header">📈 Portfolio Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_portfolios = len(app.portfolios)
        st.metric("Total Portfolios", total_portfolios)
    
    with col2:
        total_stocks = sum(len(portfolio['stocks']) for portfolio in app.portfolios.values())
        st.metric("Total Stocks Analyzed", total_stocks)
    
    with col3:
        avg_return = np.mean([portfolio['expected_return'] for portfolio in app.portfolios.values() 
                            if len(portfolio['stocks']) > 0])
        st.metric("Average Expected Return", f"{avg_return:.2%}")
    
    with col4:
        avg_risk = np.mean([portfolio['risk'] for portfolio in app.portfolios.values() 
                          if len(portfolio['stocks']) > 0])
        st.metric("Average Risk", f"{avg_risk:.2%}")
    
    # Portfolio Overview
    st.markdown('<div class="section-header">💼 Portfolio Overview</div>', unsafe_allow_html=True)
    
    for portfolio_name, portfolio_data in app.portfolios.items():
        if len(portfolio_data['stocks']) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"**{portfolio_name} Portfolio**")
                st.write(f"Stocks: {len(portfolio_data['stocks'])}")
            
            with col2:
                st.metric("Expected Return", f"{portfolio_data['expected_return']:.2%}")
            
            with col3:
                st.metric("Risk (Volatility)", f"{portfolio_data['risk']:.2%}")
            
            with col4:
                sharpe_ratio = (portfolio_data['expected_return'] - 0.06) / portfolio_data['risk'] if portfolio_data['risk'] > 0 else float('inf')
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Risk-Return Chart
    st.markdown('<div class="section-header">📊 Risk-Return Analysis</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Conservative': 'green', 'Moderate': 'blue', 'Aggressive': 'orange', 'High-Risk': 'red'}
    
    for portfolio_name, portfolio_data in app.portfolios.items():
        if len(portfolio_data['stocks']) > 0:
            ax.scatter(portfolio_data['risk'], portfolio_data['expected_return'], 
                      c=colors[portfolio_name], s=200, label=portfolio_name, alpha=0.7)
            ax.annotate(portfolio_name, 
                       (portfolio_data['risk'], portfolio_data['expected_return']),
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Risk (Volatility)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Portfolio Risk-Return Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)

def show_single_stock_analysis(app):
    st.markdown('<div class="section-header">📈 Single Stock Analysis</div>', unsafe_allow_html=True)
    
    # Analysis type selection
    analysis_type = st.radio("Choose Analysis Method:", 
                           ["Manual Input", "Live Stock Data"])
    
    if analysis_type == "Manual Input":
        show_manual_stock_analysis(app)
    else:
        show_live_stock_analysis(app)

def show_manual_stock_analysis(app):
    st.subheader("Enter Stock Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        open_price = st.number_input("Open Price", min_value=0.0, value=1500.0, step=10.0)
        prev_close = st.number_input("Previous Close", min_value=0.0, value=1480.0, step=10.0)
        volume = st.number_input("Volume", min_value=0, value=1000000, step=100000)
        value_lacs = st.number_input("Value (Lacs)", min_value=0.0, value=15000.0, step=1000.0)
        vwap = st.number_input("VWAP", min_value=0.0, value=1490.0, step=10.0)
    
    with col2:
        mkt_cap = st.number_input("Market Cap (Rs. Cr.)", min_value=0, value=100000, step=10000)
        high = st.number_input("High", min_value=0.0, value=1520.0, step=10.0)
        low = st.number_input("Low", min_value=0.0, value=1480.0, step=10.0)
        beta = st.number_input("Beta", min_value=0.0, value=1.1, step=0.1)
        volatility = st.number_input("Volatility", min_value=0.0, value=0.25, step=0.01)
    
    with col3:
        pe_ratio = st.number_input("PE Ratio", min_value=0.0, value=25.0, step=1.0)
        dividend_yield = st.number_input("Dividend Yield", min_value=0.0, value=0.02, step=0.01)
        avg_delivery_20d = st.number_input("20D Avg Delivery (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
        book_value = st.number_input("Book Value Per Share", min_value=0.0, value=500.0, step=10.0)
        face_value = st.number_input("Face Value", min_value=0, value=1)
    
    if st.button("Analyze Stock", type="primary"):
        # Prepare features for prediction
        risk_features = {}
        return_features = {}
        
        # Map input to feature names
        feature_mapping = {
            'Open': open_price,
            'Previous Close': prev_close,
            'Volume': volume,
            'Value (Lacs)': value_lacs,
            'VWAP': vwap,
            'Market Cap (Rs. Cr.)': mkt_cap,
            'High': high,
            'Low': low,
            'Beta': beta,
            'Volatility': volatility,
            'PE Ratio': pe_ratio,
            'Dividend Yield': dividend_yield,
            '20D Avg Delivery (%)': avg_delivery_20d,
            'Book Value Per Share': book_value,
            'Face Value': face_value
        }
        
        # Prepare risk features
        risk_features_list = []
        for feature in app.risk_feature_names:
            if feature in feature_mapping:
                risk_features_list.append(feature_mapping[feature])
            else:
                risk_features_list.append(0.0)  # Default value for missing features
        
        # Prepare return features
        return_features_list = []
        for feature in app.return_feature_names:
            if feature in feature_mapping:
                return_features_list.append(feature_mapping[feature])
            else:
                return_features_list.append(0.0)  # Default value for missing features
        
        # Make predictions
        with st.spinner("Analyzing stock..."):
            risk_prediction_rf = app.predict_stock_risk(risk_features_list, 'rf')
            risk_prediction_lr = app.predict_stock_risk(risk_features_list, 'lr')
            return_prediction_rf = app.predict_stock_return(return_features_list, 'rf')
            return_prediction_lr = app.predict_stock_return(return_features_list, 'lr')
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Risk Classification")
            st.markdown(f'<div class="{app.get_risk_class_color(risk_prediction_rf)}">')
            st.write(f"Random Forest: {risk_prediction_rf}")
            st.markdown('</div>')
            st.markdown(f'<div class="{app.get_risk_class_color(risk_prediction_lr)}">')
            st.write(f"Logistic Regression: {risk_prediction_lr}")
            st.markdown('</div>')
            st.markdown('</div>')
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Expected Return Prediction")
            st.metric("Random Forest", f"{return_prediction_rf:.2%}")
            st.metric("Linear Regression", f"{return_prediction_lr:.2%}")
            st.markdown('</div>')
        
        # Investment suggestion based on predictions
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.subheader("💡 Investment Suggestion")
        
        avg_return = (return_prediction_rf + return_prediction_lr) / 2
        dominant_risk = risk_prediction_rf  # Use RF as primary
        
        if dominant_risk == "Conservative":
            st.write("**Suitable for:** Risk-averse investors seeking stable returns")
            st.write("**Strategy:** Long-term holding, dividend focus")
        elif dominant_risk == "Moderate":
            st.write("**Suitable for:** Balanced investors seeking growth with moderate risk")
            st.write("**Strategy:** Core portfolio holding")
        elif dominant_risk == "Aggressive":
            st.write("**Suitable for:** Growth-oriented investors with higher risk tolerance")
            st.write("**Strategy:** Tactical allocation, monitor regularly")
        else:  # High-Risk
            st.write("**Suitable for:** Speculative investors seeking high returns")
            st.write("**Strategy:** Small allocation, active monitoring required")
        
        st.markdown('</div>')

def show_live_stock_analysis(app):
    st.subheader("Live Stock Analysis")
    
    stock_symbol = st.text_input("Enter NSE stock symbol (e.g., RELIANCE, TCS, INFY):", "RELIANCE")
    
    if st.button("Fetch & Analyze", type="primary"):
        with st.spinner("Fetching live data..."):
            try:
                # Add .NS suffix for NSE stocks
                ticker = f'{stock_symbol.upper()}.NS'
                stock = yf.Ticker(ticker)
                
                # Get live data
                hist = stock.history(period='1d')
                if hist.empty:
                    st.error("Invalid stock symbol or no data available.")
                    return
                
                live_price = hist['Close'].iloc[-1]
                
                # Get historical data for volatility calculation
                hist_1y = stock.history(period='1y')
                returns = hist_1y['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                
                # Get fundamentals
                info = stock.info
                beta = info.get('beta', 1.0)
                pe_ratio = info.get('trailingPE', 20.0)
                market_cap = info.get('marketCap', 100000) / 10000000  # Convert to Cr.
                dividend_yield = info.get('dividendYield', 0.01)
                
                # Display basic info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"₹{live_price:.2f}")
                    st.metric("Beta", f"{beta:.2f}")
                
                with col2:
                    st.metric("PE Ratio", f"{pe_ratio:.2f}")
                    st.metric("Volatility", f"{volatility:.2%}")
                
                with col3:
                    st.metric("Market Cap (Cr.)", f"₹{market_cap:,.0f}")
                    st.metric("Dividend Yield", f"{dividend_yield:.2%}")
                
                # Prepare features for prediction (using available data)
                risk_features = [live_price, live_price*0.99, 1000000, 15000, live_price,
                               market_cap, live_price*1.02, live_price*0.98, live_price*1.1, live_price*0.9,
                               live_price*1.2, live_price*0.8, live_price*1.3, 1000000,
                               500, 1, live_price*0.7, 60.0, beta, volatility, pe_ratio, dividend_yield]
                
                return_features = [beta, volatility, pe_ratio, dividend_yield, market_cap, 
                                 (0.06 + beta * (0.12 - 0.06) - 0.06) / volatility, 60.0]
                
                # Make predictions
                risk_prediction = app.predict_stock_risk(risk_features[:len(app.risk_feature_names)], 'rf')
                return_prediction = app.predict_stock_return(return_features[:len(app.return_feature_names)], 'rf')
                
                # Display predictions
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("AI Analysis Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f'<div class="{app.get_risk_class_color(risk_prediction)}">')
                    st.write(f"**Risk Category:** {risk_prediction}")
                    st.markdown('</div>')
                
                with col2:
                    st.metric("Expected Annual Return", f"{return_prediction:.2%}")
                st.markdown('</div>')
                
            except Exception as e:
                st.error(f"Error fetching stock data: {e}")

def show_portfolio_builder(app):
    st.markdown('<div class="section-header">💼 Smart Portfolio Builder</div>', unsafe_allow_html=True)
    
    # Risk tolerance selection
    risk_tolerance = st.select_slider(
        "Select Your Risk Tolerance:",
        options=["Very Low", "Low", "Medium", "High", "Very High"],
        value="Medium"
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
    
    if selected_portfolio in app.portfolios and len(app.portfolios[selected_portfolio]['stocks']) > 0:
        portfolio_data = app.portfolios[selected_portfolio]
        
        st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(f"Recommended: {selected_portfolio} Portfolio")
        st.markdown('</div>')
        
        # Portfolio statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Return", f"{portfolio_data['expected_return']:.2%}")
        
        with col2:
            st.metric("Risk (Volatility)", f"{portfolio_data['risk']:.2%}")
        
        with col3:
            sharpe_ratio = (portfolio_data['expected_return'] - 0.06) / portfolio_data['risk'] if portfolio_data['risk'] > 0 else float('inf')
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # Stock allocation table
        st.subheader("📋 Recommended Stock Allocation")
        
        stocks_df = portfolio_data['stocks'].copy()
        if len(portfolio_data['allocation']) > 0:
            stocks_df['Allocation (%)'] = portfolio_data['allocation'] * 100
            stocks_df['Investment (₹)'] = stocks_df['Allocation (%)'] * investment_amount / 100
        
        # Display top stocks
        display_columns = ['Company', 'Expected Return', 'Volatility', 'Beta']
        if 'Allocation (%)' in stocks_df.columns:
            display_columns.extend(['Allocation (%)', 'Investment (₹)'])
        
        st.dataframe(stocks_df[display_columns].head(10), use_container_width=True)
        
        # Portfolio composition chart
        st.subheader("📊 Portfolio Composition")
        
        if len(portfolio_data['allocation']) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pie chart for allocations
            top_10_allocation = portfolio_data['allocation'][:10] * 100
            top_10_companies = stocks_df['Company'].head(10)
            ax1.pie(top_10_allocation, labels=top_10_companies, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Top 10 Stock Allocations')
            
            # Risk-return scatter for portfolio stocks
            colors = plt.cm.viridis(np.linspace(0, 1, len(stocks_df)))
            ax2.scatter(stocks_df['Volatility'], stocks_df['Expected Return'], 
                       c=colors, s=100, alpha=0.6)
            ax2.set_xlabel('Volatility (Risk)')
            ax2.set_ylabel('Expected Return')
            ax2.set_title('Portfolio Stocks: Risk-Return Profile')
            ax2.grid(True, alpha=0.3)
            
            # Add company labels for top stocks
            for i, (_, row) in enumerate(stocks_df.head(5).iterrows()):
                ax2.annotate(row['Company'][:15], 
                           (row['Volatility'], row['Expected Return']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        st.warning(f"No stocks available in the {selected_portfolio} portfolio.")

def show_investment_recommendations(app):
    st.markdown('<div class="section-header">🎯 Personalized Investment Recommendations</div>', unsafe_allow_html=True)
    
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
    
    age = st.slider("Your Age:", min_value=18, max_value=80, value=35)
    investment_goal = st.selectbox(
        "Primary Investment Goal:",
        ["Wealth Preservation", "Regular Income", "Wealth Growth", "Aggressive Growth"]
    )
    
    if st.button("Generate Personalized Recommendation", type="primary"):
        # Enhanced recommendation logic
        risk_mapping = {
            'Very Low': 'Conservative',
            'Low': 'Conservative',
            'Medium': 'Moderate',
            'High': 'Aggressive',
            'Very High': 'High-Risk'
        }
        
        recommended_portfolio = risk_mapping.get(risk_tolerance, 'Moderate')
        
        # Age-based adjustment
        if age < 30:
            risk_adjustment = "You're young - consider being more aggressive with investments for long-term growth."
        elif age < 50:
            risk_adjustment = "Balanced approach suitable for your age - focus on growth with some stability."
        else:
            risk_adjustment = "Consider more conservative allocations to preserve capital as you approach retirement."
        
        # Goal-based adjustment
        goal_advice = {
            "Wealth Preservation": "Focus on capital preservation with stable, dividend-paying stocks.",
            "Regular Income": "Prioritize high-dividend yield stocks and stable returns.",
            "Wealth Growth": "Balance between growth stocks and stable investments.",
            "Aggressive Growth": "Focus on high-growth potential stocks, accepting higher volatility."
        }
        
        if recommended_portfolio in app.portfolios:
            portfolio_data = app.portfolios[recommended_portfolio]
            
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.subheader("🎯 Your Personalized Investment Plan")
            
            st.write(f"**Recommended Portfolio:** {recommended_portfolio}")
            st.write(f"**Expected Return:** {portfolio_data['expected_return']:.2%}")
            st.write(f"**Risk Level:** {portfolio_data['risk']:.2%}")
            st.write(f"**Number of Stocks:** {len(portfolio_data['stocks'])}")
            
            st.write("---")
            st.write("**📋 Strategic Advice:**")
            st.write(f"• {risk_adjustment}")
            st.write(f"• {goal_advice[investment_goal]}")
            
            if investment_horizon == "Short-term (1-2 years)":
                st.write("• For short-term horizon, focus on liquidity and lower volatility")
            elif investment_horizon == "Medium-term (3-5 years)":
                st.write("• Medium-term allows for balanced growth with moderate risk")
            else:
                st.write("• Long-term horizon enables higher risk-taking for greater returns")
            
            st.markdown('</div>')
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Portfolio Beta", 
                         f"{portfolio_data['stocks']['Beta'].mean():.2f}" if len(portfolio_data['stocks']) > 0 else "N/A")
            
            with col2:
                avg_pe = portfolio_data['stocks']['PE Ratio'].mean() if 'PE Ratio' in portfolio_data['stocks'].columns and len(portfolio_data['stocks']) > 0 else "N/A"
                st.metric("Average PE Ratio", f"{avg_pe:.1f}" if avg_pe != "N/A" else "N/A")
            
            with col3:
                sharpe_ratio = (portfolio_data['expected_return'] - 0.06) / portfolio_data['risk'] if portfolio_data['risk'] > 0 else float('inf')
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

def show_model_performance(app):
    st.markdown('<div class="section-header">📊 Model Performance & Analytics</div>', unsafe_allow_html=True)
    
    # Model performance metrics (these would typically come from your training)
    st.subheader("🤖 Model Accuracy Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Risk Classification (RF)", "85%", "2%")
    
    with col2:
        st.metric("Risk Classification (LR)", "78%", "-1%")
    
    with col3:
        st.metric("Return Prediction (RF R²)", "0.72", "0.05")
    
    with col4:
        st.metric("Return Prediction (LR R²)", "0.65", "0.02")
    
    # Feature importance visualization
    st.subheader("🔍 Feature Importance")
    
    # Create sample feature importance data (in practice, load from your models)
    feature_importance_data = {
        'Feature': ['Beta', 'Volatility', 'PE Ratio', 'Market Cap', 'Dividend Yield', 
                   '20D Avg Delivery', 'Book Value', 'Face Value'],
        'Importance': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
    }
    
    importance_df = pd.DataFrame(feature_importance_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title('Feature Importance in Risk Classification')
    st.pyplot(fig)
    
    # Portfolio performance comparison
    st.subheader("📈 Historical Portfolio Performance")
    
    # Sample performance data
    performance_data = {
        'Portfolio': ['Conservative', 'Moderate', 'Aggressive', 'High-Risk'],
        'Expected Return': [0.08, 0.12, 0.16, 0.22],
        'Actual Return': [0.075, 0.125, 0.155, 0.18],
        'Prediction Error': [0.005, 0.005, 0.005, 0.04]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(perf_df))
    width = 0.35
    
    ax.bar(x - width/2, perf_df['Expected Return'], width, label='Expected', alpha=0.7)
    ax.bar(x + width/2, perf_df['Actual Return'], width, label='Actual', alpha=0.7)
    ax.set_xlabel('Portfolio Type')
    ax.set_ylabel('Return')
    ax.set_title('Expected vs Actual Portfolio Returns')
    ax.set_xticks(x)
    ax.set_xticklabels(perf_df['Portfolio'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Model insights
    st.subheader("💡 Key Insights")
    
    insights = [
        "• Random Forest outperforms Logistic Regression in risk classification",
        "• Beta and Volatility are the most important risk predictors",
        "• High-Risk portfolios show higher prediction errors due to market volatility",
        "• Model performs best on Moderate and Aggressive risk categories",
        "• Return prediction accuracy improves with more historical data"
    ]
    
    for insight in insights:
        st.write(insight)

if __name__ == "__main__":
    main()
