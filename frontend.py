import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuration
#API_BASE_URL = "http://localhost:8000"  # Change to your deployed backend URL
API_BASE_URL = "https://ai-stock-analyst-hxdf.onrender.com"

# Page configuration
st.set_page_config(
    page_title="AI Stock Research Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stock-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        color: #000000;
    }
    .metric-card h4 {
        color: #333333;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    .metric-card p {
        color: #000000;
        margin: 0;
        font-size: 1.1rem;
        font-weight: 500;
    }
    .weather-widget {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .swot-strengths { 
        background-color: #d4edda; 
        padding: 1rem; 
        border-radius: 8px; 
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .swot-weaknesses { 
        background-color: #f8d7da; 
        padding: 1rem; 
        border-radius: 8px; 
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .swot-opportunities { 
        background-color: #d1ecf1; 
        padding: 1rem; 
        border-radius: 8px; 
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    .swot-threats { 
        background-color: #fff3cd; 
        padding: 1rem; 
        border-radius: 8px; 
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .suggestion-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .chat-follow-up {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 2rem 0;
        border-left: 4px solid #1f77b4;
        color: #000000;
    }
    .chat-follow-up h5 {
        color: #333333;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    .chat-follow-up p {
        color: #000000;
        margin: 0;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# Helper functions
def make_authenticated_request(method, endpoint, data=None):
    """Make authenticated API request"""
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=data)
        elif method == "POST":
            headers["Content-Type"] = "application/json"
            response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
        if response.status_code == 401:
            st.session_state.authenticated = False
            st.session_state.token = None
            st.error("Session expired. Please login again.")
            st.rerun()
        
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None

def login_user(email, password):
    """Login user"""
    data = {"email": email, "password": password}
    response = requests.post(f"{API_BASE_URL}/login", json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.session_state.authenticated = True
        st.session_state.token = result["token"]
        st.session_state.user = result["user"]
        return True, "Login successful!"
    else:
        return False, response.json().get("detail", "Login failed")

def register_user(email, password, full_name):
    """Register new user"""
    data = {"email": email, "password": password, "full_name": full_name}
    response = requests.post(f"{API_BASE_URL}/register", json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.session_state.authenticated = True
        st.session_state.token = result["token"]
        st.session_state.user = result["user"]
        return True, "Registration successful!"
    else:
        return False, response.json().get("detail", "Registration failed")

def get_weather_data():
    """Get weather data"""
    response = requests.get(f"{API_BASE_URL}/weather")
    if response.status_code == 200:
        return response.json()
    return {"error": "Weather data unavailable"}

def show_live_clock():
    """Show current time and date in sidebar"""
    st.markdown("### 🕐 Current Time")
    
    # Get current time
    current_time = datetime.now()
    
    # Create the clock display
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    ">
        <h3 style="margin: 0; font-size: 1.8rem;">{current_time.strftime('%H:%M:%S')}</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{current_time.strftime('%A')}</p>
        <p style="margin: 0; font-size: 1rem;">{current_time.strftime('%B %d, %Y')}</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
            Updates with app interactions
        </p>
    </div>
    """, unsafe_allow_html=True)

# Add this function for better user experience
def auto_refresh_app():
    """Add subtle auto-refresh functionality"""
    # Only refresh if no user activity for a while
    if 'last_interaction' not in st.session_state:
        st.session_state.last_interaction = time.time()
    
    # Light auto-refresh every 60 seconds when idle
    current_time = time.time()
    if current_time - st.session_state.last_interaction > 60:
        st.session_state.last_interaction = current_time
        # Minimal JavaScript refresh
        st.markdown("""
        <meta http-equiv="refresh" content="60">
        """, unsafe_allow_html=True)

def analyze_stock(symbol):
    """Analyze stock"""
    data = {"symbol": symbol, "user_id": st.session_state.user["id"]}
    response = make_authenticated_request("POST", "/analyze-stock", data)
    
    if response and response.status_code == 200:
        return response.json()
    return None

def ask_stock_question(symbol, question):
    """Ask a follow-up question about a stock"""
    data = {"symbol": symbol, "question": question, "user_id": st.session_state.user["id"]}
    response = make_authenticated_request("POST", "/ask-question", data)
    
    if response and response.status_code == 200:
        result = response.json()
        return result.get("answer", "No answer received.")
    return None

def ask_general_finance_question(question):
    """Ask a general finance question"""
    try:
        data = {"question": question, "user_id": st.session_state.user["id"]}
        response = make_authenticated_request("POST", "/ask-general", data)
        
        if response and response.status_code == 200:
            result = response.json()
            return result.get("answer", "No answer received.")
        elif response:
            print(f"API Error: Status {response.status_code}, Response: {response.text}")
            return None
        else:
            print("No response received from API")
            return None
    except Exception as e:
        print(f"Error in ask_general_finance_question: {str(e)}")
        return None

# Authentication UI
def show_auth_ui():
    """Show authentication interface"""
    st.markdown('<h1 class="main-header">🤖 AI Stock Research Analyst</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.subheader("Login to Your Account")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", use_container_width=True):
                if email and password:
                    success, message = login_user(email, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill in all fields")
        
        with tab2:
            st.subheader("Create New Account")
            full_name = st.text_input("Full Name", key="register_name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm")
            
            if st.button("Sign Up", use_container_width=True):
                if full_name and email and password and confirm_password:
                    if password == confirm_password:
                        success, message = register_user(email, password, full_name)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")

# Weather widget
def show_weather_widget():
    """Show weather widget in sidebar"""
    st.markdown("### 🌤️ Current Weather")
    
    weather_data = get_weather_data()
    
    if "error" not in weather_data:
        st.markdown(f"""
        <div class="weather-widget">
            <h4>{weather_data.get('city', 'Unknown')}</h4>
            <p style="font-size: 2rem; margin: 0;">{weather_data.get('temperature', 0):.1f}°C</p>
            <p style="margin: 0;">{weather_data.get('description', '').title()}</p>
            <p style="margin: 0; font-size: 0.9rem;">Humidity: {weather_data.get('humidity', 0)}%</p>
            <p style="margin: 0; font-size: 0.9rem;">Wind: {weather_data.get('wind_speed', 0)} m/s</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Weather data unavailable")

# Stock analysis display
def show_stock_analysis(analysis_data):
    """Display comprehensive stock analysis"""
    stock_data = analysis_data.get('stock_data', {})
    analysis = analysis_data.get('analysis', '')
    swot = analysis_data.get('swot_analysis', {})
    suggested_stocks = analysis_data.get('suggested_stocks', [])
    chart_b64 = analysis_data.get('chart', '')
    
    # Stock overview card
    st.markdown(f"""
    <div class="stock-card">
        <h2>{stock_data.get('company_name', 'N/A')} ({stock_data.get('symbol', 'N/A')})</h2>
        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
            <div>
                <h3>${stock_data.get('current_price', 0):.2f}</h3>
                <p>Current Price</p>
            </div>
            <div>
                <h3 style="color: {'#00ff00' if stock_data.get('price_change_pct', 0) >= 0 else '#ff4444'};">
                    {stock_data.get('price_change_pct', 0):+.2f}%
                </h3>
                <p>24h Change</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Market Cap</h4>
            <p>${stock_data.get('market_cap', 0):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>P/E Ratio</h4>
            <p>{stock_data.get('pe_ratio', 0):.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Volume</h4>
            <p>{stock_data.get('volume', 0):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Sector</h4>
            <p>{stock_data.get('sector', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Chart
    if chart_b64:
        st.subheader("📊 Price Chart")
        try:
            import base64
            chart_image = base64.b64decode(chart_b64)
            st.image(chart_image, use_container_width=True)
        except Exception as e:
            st.info("📊 Chart visualization temporarily unavailable")
    
    # AI Analysis
    st.subheader("🤖 AI Analysis")
    st.markdown(analysis)
    
    # SWOT Analysis
    st.subheader("🎯 SWOT Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Strengths")
        strengths_html = '<ul>' + ''.join([f'<li>{item}</li>' for item in swot.get('strengths', [])]) + '</ul>'
        st.markdown(f"""
        <div class="swot-strengths">
            {strengths_html}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Opportunities")
        opportunities_html = '<ul>' + ''.join([f'<li>{item}</li>' for item in swot.get('opportunities', [])]) + '</ul>'
        st.markdown(f"""
        <div class="swot-opportunities">
            {opportunities_html}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Weaknesses")
        weaknesses_html = '<ul>' + ''.join([f'<li>{item}</li>' for item in swot.get('weaknesses', [])]) + '</ul>'
        st.markdown(f"""
        <div class="swot-weaknesses">
            {weaknesses_html}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Threats")
        threats_html = '<ul>' + ''.join([f'<li>{item}</li>' for item in swot.get('threats', [])]) + '</ul>'
        st.markdown(f"""
        <div class="swot-threats">
            {threats_html}
        </div>
        """, unsafe_allow_html=True)
    
    # Suggested Stocks
    if suggested_stocks:
        st.subheader("💡 You Might Also Like")
        st.markdown(f"""
        <div class="suggestion-box">
            <h4>🌟 Suggested Stocks Based on Your Analysis</h4>
            <p>Investors who analyzed {stock_data.get('symbol', 'N/A')} also looked at these stocks:</p>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(len(suggested_stocks))
        for i, symbol in enumerate(suggested_stocks):
            with cols[i]:
                if st.button(f"📈 {symbol}", use_container_width=True, key=f"suggested_{symbol}"):
                    with st.spinner(f"🔄 Analyzing {symbol}..."):
                        new_analysis = analyze_stock(symbol)
                        if new_analysis:
                            st.session_state.current_analysis = new_analysis
                            st.session_state.last_searched = symbol
                            st.rerun()
    
    # Follow-up Questions Section
    st.markdown("---")
    st.subheader("💬 Ask More About This Stock")
    st.markdown(f"""
    <div class="chat-follow-up">
        <h5>🎯 Stock-Specific Questions for {stock_data.get('symbol', 'N/A')}</h5>
        <p>Ask detailed questions about this specific stock's performance, outlook, earnings, or any aspect of {stock_data.get('company_name', 'this company')}.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Question input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input(
            f"Ask specifically about {stock_data.get('symbol', 'this stock')}:",
            placeholder=f"e.g., 'What are {stock_data.get('symbol', 'AAPL')}'s latest earnings results?'",
            key="follow_up_question"
        )
    with col2:
        ask_button = st.button("🚀 Ask", use_container_width=True)
    
    # Example questions
    st.markdown(f"**💡 Example questions about {stock_data.get('symbol', 'this stock')}:**")
    example_questions = [
        f"What are the latest earnings for {stock_data.get('symbol', 'N/A')}?",
        f"How does {stock_data.get('symbol', 'N/A')} compare to its competitors?",
        f"What is the outlook for {stock_data.get('symbol', 'N/A')} next quarter?",
        f"Any recent news about {stock_data.get('symbol', 'N/A')}?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(question, key=f"example_q_{i}", use_container_width=True):
                # Directly process the question instead of modifying session state
                with st.spinner("🔍 Getting answer..."):
                    answer = ask_stock_question(stock_data.get('symbol', 'N/A'), question)
                    if answer:
                        st.success("✅ Answer received!")
                        st.markdown("### 📝 Answer:")
                        st.markdown(answer)
                    else:
                        st.error("❌ Sorry, couldn't get an answer. Please try again.")
    
    # Handle question submission
    if ask_button and user_question:
        with st.spinner("🔍 Getting answer..."):
            answer = ask_stock_question(stock_data.get('symbol', 'N/A'), user_question)
            if answer:
                st.success("✅ Answer received!")
                st.markdown("### 📝 Answer:")
                st.markdown(answer)
            else:
                st.error("❌ Sorry, couldn't get an answer. Please try again.")
    elif ask_button and not user_question:
        st.warning("⚠️ Please enter a question first.")

# Main application
def show_main_app():
    """Show main application interface"""
    # Add subtle auto-refresh
    auto_refresh_app()
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state.user['full_name']}!")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()
        
        st.markdown("---")
        
        # Live Clock and Date
        show_live_clock()
        
        st.markdown("---")
        
        # Weather widget
        show_weather_widget()
    
    # Main content
    st.markdown('<h1 class="main-header">📈 AI Stock Research Analyst</h1>', unsafe_allow_html=True)
    
    # Financial Assistant Chat at the top
    st.subheader("🤖 Your AI Financial Assistant")
    st.markdown("""
    <div class="chat-follow-up">
        <h5>💡 Ask me anything about finance, investing, or markets!</h5>
        <p>I can help with market analysis, investment strategies, economic trends, or specific stock questions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # General finance question input
    col1, col2 = st.columns([4, 1])
    with col1:
        general_question = st.text_input(
            "Ask your financial question:",
            placeholder="e.g., 'What's the current market outlook?' or 'How should I diversify my portfolio?'",
            key="general_finance_question"
        )
    with col2:
        general_ask_button = st.button("💬 Ask", use_container_width=True)
    
    # Handle general question
    if general_ask_button and general_question:
        with st.spinner("🔍 Analyzing financial markets..."):
            answer = ask_general_finance_question(general_question)
            if answer:
                st.success("✅ Here's your financial analysis!")
                st.markdown("### 📝 Financial Insight:")
                st.markdown(answer)
            else:
                st.error("❌ Sorry, couldn't get an answer. Please try again.")
    elif general_ask_button and not general_question:
        st.warning("⚠️ Please enter a financial question first.")
    
    # Example financial questions
    st.markdown("**💡 Try these popular questions:**")
    finance_examples = [
        "What's the current market outlook for 2025?",
        "How should I diversify my investment portfolio?",
        "What are the best sectors to invest in right now?",
        "How do interest rates affect stock prices?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(finance_examples):
        with cols[i % 2]:
            if st.button(question, key=f"finance_example_{i}", use_container_width=True):
                # Directly process the question instead of modifying session state
                with st.spinner("🔍 Analyzing financial markets..."):
                    answer = ask_general_finance_question(question)
                    if answer:
                        st.success("✅ Here's your financial analysis!")
                        st.markdown("### 📝 Financial Insight:")
                        st.markdown(answer)
                    else:
                        st.error("❌ Sorry, couldn't get an answer. Please try again.")
    
    st.markdown("---")
    
    # Stock analysis section
    st.subheader("🔍 Analyze Any Stock")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_symbol = st.text_input(
            "Enter Stock Symbol (e.g., AAPL, GOOGL, TSLA)",
            placeholder="Enter stock symbol...",
            key="stock_search"
        )
    
    with col2:
        search_button = st.button("🚀 Analyze", use_container_width=True, type="primary")
    
    # Popular stocks quick access
    st.markdown("#### 🌟 Popular Stocks")
    popular_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    
    cols = st.columns(len(popular_stocks))
    for i, symbol in enumerate(popular_stocks):
        with cols[i]:
            if st.button(symbol, use_container_width=True, key=f"popular_{symbol}"):
                # Trigger analysis directly instead of modifying the input
                with st.spinner(f"🔄 Analyzing {symbol}... This may take a moment."):
                    analysis_data = analyze_stock(symbol)
                    
                    if analysis_data:
                        st.session_state.current_analysis = analysis_data
                        st.session_state.last_searched = symbol
                        st.success(f"✅ Analysis complete for {symbol}!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to analyze stock. Please try again.")
    
    # Analyze stock
    if search_button and stock_symbol:
        with st.spinner(f"🔄 Analyzing {stock_symbol.upper()}... This may take a moment."):
            analysis_data = analyze_stock(stock_symbol.upper())
            
            if analysis_data:
                st.session_state.current_analysis = analysis_data
                st.session_state.last_searched = stock_symbol.upper()
                st.success(f"✅ Analysis complete for {stock_symbol.upper()}!")
            else:
                st.error("❌ Failed to analyze stock. Please check the symbol and try again.")
    elif search_button and not stock_symbol:
        st.warning("⚠️ Please enter a stock symbol")
    
    # Display current analysis
    if st.session_state.current_analysis:
        st.markdown("---")
        show_stock_analysis(st.session_state.current_analysis)

# Main app logic
def main():
    """Main application logic"""
    if not st.session_state.authenticated:
        show_auth_ui()
    else:
        show_main_app()

if __name__ == "__main__":
    main()