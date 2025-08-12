from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import httpx
import json
import os
from datetime import datetime, timedelta
import jwt
import bcrypt
from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai
import yfinance as yf
import plotly.graph_objs as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import base64
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "your_supabase_url")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your_supabase_key")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "your_weather_api_key")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_tavily_api_key")
JWT_SECRET = os.getenv("JWT_SECRET", "your_jwt_secret_key")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "your_alpha_vantage_key")

# Initialize services
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)
security = HTTPBearer()

# Pydantic models
class UserRegister(BaseModel):
    email: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: str
    password: str

class StockQuery(BaseModel):
    symbol: str
    user_id: str

class ChatMessage(BaseModel):
    user_id: str
    query: str
    response: str
    symbol: Optional[str] = None

class StockQuestion(BaseModel):
    symbol: str
    question: str
    user_id: str

class GeneralFinanceQuestion(BaseModel):
    question: str
    user_id: str

# Database initialization
async def init_db():
    """Initialize database tables if they don't exist"""
    try:
        # Users table
        await supabase.table('users').select("*").limit(1).execute()
    except:
        # Create users table if it doesn't exist
        print("Users table might not exist, ensure it's created in Supabase")
    
    try:
        # Chat history table
        await supabase.table('chat_history').select("*").limit(1).execute()
    except:
        print("Chat history table might not exist, ensure it's created in Supabase")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown
    pass

# FastAPI app
app = FastAPI(title="Stock Research Analyst API", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_jwt_token(user_id: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_jwt_token(token: str) -> Dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_jwt_token(token)
    return payload

# Stock data functions
async def get_stock_data(symbol: str) -> Dict:
    """Get comprehensive stock data"""
    try:
        stock = yf.Ticker(symbol)
        
        # Get basic info
        info = stock.info
        
        # Get historical data
        hist = stock.history(period="1y")
        
        if hist.empty:
            raise HTTPException(status_code=400, detail=f"No data found for symbol {symbol}")
        
        # # Get financial data (safely)
        # try:
        #     financials = stock.financials
        #     balance_sheet = stock.balance_sheet
        # except:
        #     financials = None
        #     balance_sheet = None
        
        # Calculate technical indicators (with safe conversion)
        current_price = float(hist['Close'].iloc[-1]) if len(hist) > 0 else 0.0
        price_change = float(current_price - hist['Close'].iloc[-2]) if len(hist) > 1 else 0.0
        price_change_pct = (price_change / float(hist['Close'].iloc[-2]) * 100) if len(hist) > 1 and hist['Close'].iloc[-2] != 0 else 0.0
        
        # Moving averages (safely)
        ma_50 = float(hist['Close'].rolling(window=50).mean().iloc[-1]) if len(hist) >= 50 else current_price
        ma_200 = float(hist['Close'].rolling(window=200).mean().iloc[-1]) if len(hist) >= 200 else current_price
        
        # Convert historical data to serializable format
        historical_data = {
            'Close': {str(date): float(price) for date, price in hist['Close'].items()},
            'Volume': {str(date): int(vol) for date, vol in hist['Volume'].items()},
            'High': {str(date): float(price) for date, price in hist['High'].items()},
            'Low': {str(date): float(price) for date, price in hist['Low'].items()},
            'Open': {str(date): float(price) for date, price in hist['Open'].items()}
        }
        
        return {
            "symbol": symbol,
            "company_name": info.get("longName", symbol),
            "current_price": current_price,
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "market_cap": int(info.get("marketCap", 0)) if info.get("marketCap") else 0,
            "volume": int(hist['Volume'].iloc[-1]) if len(hist) > 0 else 0,
            "pe_ratio": float(info.get("trailingPE", 0)) if info.get("trailingPE") else 0.0,
            "dividend_yield": float(info.get("dividendYield", 0)) if info.get("dividendYield") else 0.0,
            "ma_50": ma_50,
            "ma_200": ma_200,
            "sector": str(info.get("sector", "Unknown")),
            "industry": str(info.get("industry", "Unknown")),
            "historical_data": historical_data,
            "info": {k: str(v) if v is not None else "" for k, v in info.items()}  # Convert all to strings
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching stock data: {str(e)}")

async def generate_stock_analysis(stock_data: Dict) -> str:
    """Generate AI-powered stock analysis using Gemini"""
    try:
        # Initialize Gemini model with the correct model name
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a professional stock analyst. Provide a comprehensive analysis of {stock_data['company_name']} ({stock_data['symbol']}).
        
        IMPORTANT GUIDELINES:
        - Only provide stock and financial analysis
        - Do not answer questions unrelated to finance, stocks, or investing
        - Focus on data-driven insights
        - Be professional and objective
        
        Current Data:
        - Current Price: ${stock_data['current_price']:.2f}
        - Price Change: {stock_data['price_change_pct']:.2f}%
        - Market Cap: ${stock_data['market_cap']:,}
        - P/E Ratio: {stock_data['pe_ratio']}
        - Sector: {stock_data['sector']}
        - Industry: {stock_data['industry']}
        - 50-day MA: ${stock_data['ma_50']:.2f}
        - 200-day MA: ${stock_data['ma_200']:.2f}
        
        Please provide:
        1. Company Overview
        2. Financial Health Assessment
        3. Technical Analysis
        4. Market Position
        5. Investment Recommendation (Buy/Hold/Sell)
        6. Risk Factors
        7. Price Target (if applicable)
        
        Keep the analysis professional, data-driven, and actionable. Limit response to 1000 words.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

async def generate_swot_analysis(stock_data: Dict) -> Dict:
    """Generate SWOT analysis using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Create a detailed SWOT analysis for {stock_data['company_name']} ({stock_data['symbol']}).
        
        Company Information:
        - Sector: {stock_data['sector']}
        - Industry: {stock_data['industry']}
        - Market Cap: ${stock_data['market_cap']:,}
        - Current Performance: {stock_data['price_change_pct']:.2f}% change
        
        Provide a structured SWOT analysis with exactly 4 points each:
        
        Return ONLY a valid JSON object with this exact format:
        {{
            "strengths": ["point 1", "point 2", "point 3", "point 4"],
            "weaknesses": ["point 1", "point 2", "point 3", "point 4"],
            "opportunities": ["point 1", "point 2", "point 3", "point 4"],
            "threats": ["point 1", "point 2", "point 3", "point 4"]
        }}
        
        Each point should be concise (1-2 sentences maximum).
        """
        
        response = model.generate_content(prompt)
        
        # Try to parse as JSON, fallback to structured text
        try:
            # Clean the response text
            response_text = response.text.strip()
            # Remove any markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            swot_data = json.loads(response_text)
        except:
            # Fallback parsing
            swot_data = {
                "strengths": ["Strong market position", "Solid financial performance", "Innovation capability", "Brand recognition"],
                "weaknesses": ["Market competition", "Regulatory challenges", "Operational costs", "Market dependency"],
                "opportunities": ["Market expansion", "Technology adoption", "Strategic partnerships", "Product diversification"],
                "threats": ["Economic downturn", "Increased competition", "Regulatory changes", "Market volatility"]
            }
        
        return swot_data
        
    except Exception as e:
        return {
            "strengths": ["Analysis unavailable due to technical error"],
            "weaknesses": ["Analysis unavailable due to technical error"],
            "opportunities": ["Analysis unavailable due to technical error"],
            "threats": ["Analysis unavailable due to technical error"]
        }

async def generate_suggested_stocks(stock_data: Dict) -> List[str]:
    """Generate suggested stocks based on current stock analysis"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Based on the analysis of {stock_data['company_name']} ({stock_data['symbol']}) in the {stock_data['sector']} sector,
        suggest 5 other stocks that investors might be interested in.
        
        Consider:
        - Same sector/industry competitors
        - Similar market cap companies
        - Complementary investments
        - Growth potential
        
        Return only a JSON array of stock symbols (no company names):
        ["SYMBOL1", "SYMBOL2", "SYMBOL3", "SYMBOL4", "SYMBOL5"]
        """
        
        response = model.generate_content(prompt)
        
        try:
            # Clean and parse response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            suggestions = json.loads(response_text)
            return suggestions if isinstance(suggestions, list) else []
        except:
            # Fallback suggestions based on sector
            sector_suggestions = {
                "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
                "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
                "Financial": ["JPM", "BAC", "WFC", "GS", "MS"],
                "Consumer": ["KO", "PEP", "WMT", "HD", "MCD"],
                "Energy": ["XOM", "CVX", "COP", "EOG", "SLB"]
            }
            return sector_suggestions.get(stock_data.get('sector', 'Technology'), ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
            
    except Exception as e:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]  # Default suggestions

async def get_real_time_stock_info(symbol: str, query: str) -> str:
    """Get real-time information about a stock using Tavily API"""
    try:
        async with httpx.AsyncClient() as client:
            url = "https://api.tavily.com/search"
            headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
            
            # Enhanced search query for better results
            search_queries = [
                f"{symbol} stock {query} latest news 2025",
                f"{symbol} earnings financial results recent",
                f"{symbol} stock price target analyst recommendations"
            ]
            
            all_info = ""
            
            for search_query in search_queries:
                data = {
                    "query": search_query,
                    "search_depth": "advanced",
                    "max_results": 5,
                    "include_answer": True
                }
                
                response = await client.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    # Include the AI answer if available
                    if results.get("answer"):
                        all_info += f"Key Information: {results['answer']}\n\n"
                    
                    # Include search results
                    if results.get("results"):
                        for result in results["results"][:3]:
                            title = result.get('title', '')
                            content = result.get('content', '')
                            url = result.get('url', '')
                            
                            all_info += f"📰 {title}\n"
                            all_info += f"📍 Source: {url}\n"
                            all_info += f"📝 {content[:300]}...\n\n"
                    
                    # Only do one search that returns good results
                    if all_info.strip():
                        break
            
            return all_info if all_info.strip() else "No recent information found."
                
    except Exception as e:
        return f"Error fetching real-time data: {str(e)}"

async def answer_stock_question(symbol: str, question: str, stock_data: Dict) -> str:
    """Answer follow-up questions about a stock with enhanced domain focus"""
    try:
        # Check if question is finance-related
        finance_keywords = [
            'stock', 'price', 'earnings', 'revenue', 'profit', 'loss', 'dividend', 'market', 'investment', 
            'portfolio', 'trading', 'analysis', 'forecast', 'valuation', 'financial', 'quarterly',
            'annual', 'report', 'growth', 'decline', 'performance', 'competitor', 'sector', 'industry',
            'buy', 'sell', 'hold', 'recommendation', 'target', 'outlook', 'risk', 'volatility',
            'PE ratio', 'market cap', 'volume', 'technical', 'fundamental', 'analyst', 'rating'
        ]
        
        question_lower = question.lower()
        is_finance_related = any(keyword in question_lower for keyword in finance_keywords)
        
        if not is_finance_related:
            return f"I'm your AI Stock Research Assistant. I can only help with questions about {symbol} or other financial and investment topics. Please ask about stock performance, earnings, market analysis, investment recommendations, or any finance-related questions."
        
        # Get comprehensive real-time info
        print(f"Fetching real-time info for {symbol} with query: {question}")
        real_time_info = await get_real_time_stock_info(symbol, question)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an expert AI Stock Research Assistant specializing in financial analysis and investment guidance.
        
        Current Question: {question}
        Stock Symbol: {symbol}
        
        === CURRENT STOCK DATA ===
        Company: {stock_data['company_name']}
        Current Price: ${stock_data['current_price']:.2f}
        Price Change: {stock_data['price_change_pct']:.2f}%
        Market Cap: ${stock_data['market_cap']:,}
        P/E Ratio: {stock_data['pe_ratio']}
        Sector: {stock_data['sector']}
        Industry: {stock_data['industry']}
        
        === LATEST MARKET INFORMATION ===
        {real_time_info}
        
        === INSTRUCTIONS ===
        1. ONLY answer questions related to stocks, finance, investing, and markets
        2. Use both the current stock data AND the latest market information
        3. Provide actionable insights and professional analysis
        4. If asked about non-financial topics, politely redirect to stock/finance questions
        5. Be conversational but maintain professional expertise
        6. Include specific data points and numbers when available
        7. Mention sources of recent information when relevant
        8. Keep responses focused and under 400 words
        
        === RESPONSE GUIDELINES ===
        - Start with a direct answer to their question
        - Include relevant recent news/information
        - Provide context using current stock metrics
        - End with actionable insights or follow-up suggestions
        - Use bullet points for clarity when listing multiple points
        
        Answer the question as an expert financial assistant:
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"I apologize, but I encountered an error while analyzing {symbol}. Please try asking your question again or ask about a different aspect of this stock."

async def answer_general_finance_question(question: str) -> str:
    """Answer general finance and investment questions with real-time data"""
    try:
        print(f"Processing general finance question: {question}")
        
        # Check if question is finance-related
        finance_keywords = [
            'stock', 'investment', 'market', 'trading', 'portfolio', 'finance', 'economic', 'economy',
            'inflation', 'interest rate', 'fed', 'earnings', 'dividend', 'bond', 'mutual fund',
            'etf', 'crypto', 'cryptocurrency', 'nasdaq', 'dow', 's&p', 'forex', 'commodity',
            'recession', 'bull market', 'bear market', 'volatility', 'risk', 'return', 'diversify',
            'sectors', 'outlook', 'prices', 'affect'
        ]
        
        question_lower = question.lower()
        is_finance_related = any(keyword in question_lower for keyword in finance_keywords)
        
        if not is_finance_related:
            return "I'm your AI Financial Assistant. I specialize in stocks, investments, market analysis, and financial guidance. Please ask me about financial markets, investment strategies, economic trends, or specific stocks."
        
        # Try to get real-time financial information with fallback
        real_time_info = "Using latest available financial knowledge."
        
        try:
            if TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key":
                async with httpx.AsyncClient(timeout=10.0) as client:
                    url = "https://api.tavily.com/search"
                    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
                    data = {
                        "query": f"{question} financial markets latest 2025",
                        "search_depth": "basic",
                        "max_results": 3,
                        "include_answer": True
                    }
                    
                    response = await client.post(url, headers=headers, json=data)
                    
                    if response.status_code == 200:
                        results = response.json()
                        info_parts = []
                        
                        if results.get("answer"):
                            info_parts.append(f"Market Insight: {results['answer']}")
                        
                        if results.get("results"):
                            for result in results["results"][:2]:
                                title = result.get('title', '')
                                content = result.get('content', '')
                                if title and content:
                                    info_parts.append(f"📊 {title}: {content[:150]}...")
                        
                        if info_parts:
                            real_time_info = "\n\n".join(info_parts)
                        
                        print("Tavily API call successful")
                    else:
                        print(f"Tavily API returned status: {response.status_code}")
            else:
                print("Tavily API key not configured, using fallback")
                
        except Exception as tavily_error:
            print(f"Tavily API error: {str(tavily_error)}")
            # Continue with fallback
        
        print("Generating AI response...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an expert AI Financial Assistant and Investment Advisor.
        
        User Question: {question}
        
        === MARKET CONTEXT ===
        {real_time_info}
        
        === YOUR EXPERTISE ===
        - Professional financial advisor with deep market knowledge
        - Provide actionable investment guidance and market analysis
        - Use current financial principles and market trends
        - Be conversational yet authoritative
        
        === RESPONSE GUIDELINES ===
        1. ONLY discuss finance, investing, markets, and economics
        2. Provide practical, actionable advice
        3. Include specific strategies or recommendations when appropriate
        4. Use professional financial terminology appropriately
        5. Keep responses comprehensive yet concise (300-500 words)
        6. Structure your response clearly with key points
        7. If recent market data is available, reference it
        
        === RESPONSE STRUCTURE ===
        - Direct answer to the question
        - Supporting analysis or reasoning
        - Actionable recommendations or insights
        - Any relevant market considerations
        
        Provide your expert financial analysis:
        """
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            print("AI response generated successfully")
            return response.text
        else:
            print("AI response was empty")
            return "I apologize, but I couldn't generate a comprehensive response at the moment. Please try rephrasing your question or ask about a specific financial topic like market trends, investment strategies, or portfolio diversification."
        
    except Exception as e:
        print(f"Error in answer_general_finance_question: {str(e)}")
        return f"I encountered a technical issue while processing your financial question. Please try asking about a specific financial topic like market outlook, investment strategies, or portfolio management."

async def get_weather_data(city: str = "New York") -> Dict:
    """Get current weather data"""
    try:
        async with httpx.AsyncClient() as client:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "city": data["name"],
                    "temperature": data["main"]["temp"],
                    "description": data["weather"][0]["description"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"]
                }
            else:
                return {"error": "Weather data unavailable"}
    except Exception as e:
        return {"error": f"Weather API error: {str(e)}"}

def create_stock_chart(stock_data: Dict) -> str:
    """Create stock price chart and return base64 encoded image"""
    try:
        hist_data = stock_data['historical_data']
        dates = list(hist_data['Close'].keys())
        prices = list(hist_data['Close'].values())
        
        # Ensure we have data
        if not dates or not prices:
            return ""
        
        fig = go.Figure(data=go.Scatter(x=dates, y=prices, mode='lines', name='Stock Price'))
        fig.update_layout(
            title=f"{stock_data['company_name']} Stock Price (1 Year)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            width=800,
            height=400
        )
        
        # Convert to base64
        img_bytes = fig.to_image(format="png")
        img_base64 = base64.b64encode(img_bytes).decode()
        return img_base64
        
    except Exception as e:
        print(f"Chart creation error: {str(e)}")
        return ""

# def generate_pdf_report(stock_data: Dict, analysis: str, swot: Dict) -> str:
#     """Generate PDF report and return base64 encoded"""
#     try:
#         buffer = BytesIO()
#         p = canvas.Canvas(buffer, pagesize=letter)
        
#         # Title
#         p.setFont("Helvetica-Bold", 16)
#         p.drawString(50, 750, f"Stock Analysis Report: {stock_data['company_name']}")
        
#         # Basic info
#         p.setFont("Helvetica", 12)
#         y_position = 720
        
#         info_lines = [
#             f"Symbol: {stock_data['symbol']}",
#             f"Current Price: ${stock_data['current_price']:.2f}",
#             f"Price Change: {stock_data['price_change_pct']:.2f}%",
#             f"Market Cap: ${stock_data['market_cap']:,}",
#             f"Sector: {stock_data['sector']}",
#             f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#         ]
        
#         for line in info_lines:
#             p.drawString(50, y_position, line)
#             y_position -= 20
        
#         # Analysis section
#         y_position -= 20
#         p.setFont("Helvetica-Bold", 14)
#         p.drawString(50, y_position, "Analysis Summary:")
        
#         y_position -= 20
#         p.setFont("Helvetica", 10)
        
#         # Split analysis into lines
#         analysis_lines = analysis[:1000].split('\n')  # Limit for PDF
#         for line in analysis_lines[:20]:  # First 20 lines
#             if y_position < 100:
#                 p.showPage()
#                 y_position = 750
#             p.drawString(50, y_position, line[:80])  # Limit line length
#             y_position -= 15
        
#         p.save()
#         buffer.seek(0)
#         pdf_base64 = base64.b64encode(buffer.getvalue()).decode()
#         buffer.close()
        
#         return pdf_base64
        
#     except Exception as e:
#         return ""

# API Routes

@app.post("/register")
async def register_user(user_data: UserRegister):
    """Register a new user"""
    try:
        # Check if user exists
        existing_user = supabase.table('users').select("*").eq('email', user_data.email).execute()
        
        if existing_user.data:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash password and create user
        hashed_password = hash_password(user_data.password)
        
        result = supabase.table('users').insert({
            "email": user_data.email,
            "password_hash": hashed_password,
            "full_name": user_data.full_name,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        if result.data:
            user = result.data[0]
            token = create_jwt_token(str(user['id']), user['email'])
            
            return {
                "message": "User registered successfully",
                "token": token,
                "user": {
                    "id": user['id'],
                    "email": user['email'],
                    "full_name": user['full_name']
                }
            }
        else:
            raise HTTPException(status_code=400, detail="Registration failed")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask-general")
async def ask_general_finance_question(question_data: GeneralFinanceQuestion, current_user: Dict = Depends(get_current_user)):
    """Ask a general finance or investment question"""
    try:
        print(f"General finance question received: {question_data.question}")
        print(f"User ID: {current_user.get('user_id', 'Unknown')}")
        
        # Generate answer using real-time data
        answer = await answer_general_finance_question(question_data.question)
        
        if not answer or answer.strip() == "":
            print("Empty answer generated")
            answer = "I apologize, but I couldn't generate a response at the moment. Please try asking about specific financial topics like market trends, investment strategies, or economic outlook."
        
        print(f"Answer generated successfully, length: {len(answer)}")
        
        # Save to chat history
        try:
            chat_data = {
                "user_id": current_user["user_id"],
                "query": question_data.question,
                "response": answer[:500] + "..." if len(answer) > 500 else answer,
                "symbol": None,  # No specific symbol for general questions
                "created_at": datetime.utcnow().isoformat()
            }
            
            supabase.table('chat_history').insert(chat_data).execute()
            print("Chat history saved successfully")
        except Exception as chat_error:
            print(f"Chat history save error: {str(chat_error)}")
            # Don't fail the request if chat history fails
        
        return {
            "question": question_data.question,
            "answer": answer
        }
        
    except HTTPException as he:
        print(f"HTTP Exception in ask_general_finance_question: {str(he)}")
        raise he
    except Exception as e:
        print(f"Unexpected error in ask_general_finance_question: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail="I encountered an error while processing your financial question. Please try asking about market trends, investment strategies, or specific financial topics."
        )

@app.post("/login")
async def login_user(user_data: UserLogin):
    """Login user"""
    try:
        # Get user from database
        result = supabase.table('users').select("*").eq('email', user_data.email).execute()
        
        if not result.data:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user = result.data[0]
        
        # Verify password
        if not verify_password(user_data.password, user['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create JWT token
        token = create_jwt_token(str(user['id']), user['email'])
        
        return {
            "message": "Login successful",
            "token": token,
            "user": {
                "id": user['id'],
                "email": user['email'],
                "full_name": user['full_name']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/weather")
async def get_weather(city: str = "New York"):
    """Get current weather data"""
    weather_data = await get_weather_data(city)
    return weather_data

@app.post("/analyze-stock")
async def analyze_stock(query: StockQuery, current_user: Dict = Depends(get_current_user)):
    """Analyze a stock and return comprehensive data"""
    try:
        print(f"Analyzing stock: {query.symbol}")
        
        # Get stock data
        stock_data = await get_stock_data(query.symbol)
        print("Stock data retrieved successfully")
        
        # Generate AI analysis
        analysis = await generate_stock_analysis(stock_data)
        print("AI analysis generated successfully")
        
        # Generate SWOT analysis
        swot = await generate_swot_analysis(stock_data)
        print("SWOT analysis generated successfully")
        
        # Generate suggested stocks
        suggested_stocks = await generate_suggested_stocks(stock_data)
        print("Suggested stocks generated successfully")
        
        # Create chart
        chart_b64 = create_stock_chart(stock_data)
        print("Chart created successfully")
        
        # Generate PDF report
        # pdf_b64 = generate_pdf_report(stock_data, analysis, swot)
        # print("PDF report generated successfully")
        
        # Save to chat history
        try:
            chat_data = {
                "user_id": current_user["user_id"],
                "query": f"Analyze {query.symbol}",
                "response": analysis[:500] + "..." if len(analysis) > 500 else analysis,
                "symbol": query.symbol,
                "created_at": datetime.utcnow().isoformat()
            }
            
            supabase.table('chat_history').insert(chat_data).execute()
            print("Chat history saved successfully")
        except Exception as chat_error:
            print(f"Chat history save error: {str(chat_error)}")
            # Don't fail the whole request if chat history fails
        
        return {
            "stock_data": stock_data,
            "analysis": analysis,
            "swot_analysis": swot,
            "suggested_stocks": suggested_stocks,
            "chart": chart_b64
            #"pdf_report": pdf_b64
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_stock: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# @app.get("/chat-history")
# async def get_chat_history(current_user: Dict = Depends(get_current_user)):
#     """Get user's chat history"""
#     try:
#         result = supabase.table('chat_history').select("*").eq('user_id', current_user["user_id"]).order('created_at', desc=True).limit(50).execute()
        
#         return {"chat_history": result.data}
        
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.delete("/chat-history/{chat_id}")
# async def delete_chat(chat_id: str, current_user: Dict = Depends(get_current_user)):
#     """Delete a chat from history"""
#     try:
#         result = supabase.table('chat_history').delete().eq('id', chat_id).eq('user_id', current_user["user_id"]).execute()
        
#         return {"message": "Chat deleted successfully"}
        
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask-question")
async def ask_stock_question(question_data: StockQuestion, current_user: Dict = Depends(get_current_user)):
    """Ask a follow-up question about a stock"""
    try:
        # Get stock data for context
        stock_data = await get_stock_data(question_data.symbol)
        
        # Generate answer
        answer = await answer_stock_question(question_data.symbol, question_data.question, stock_data)
        
        # Save to chat history
        try:
            chat_data = {
                "user_id": current_user["user_id"],
                "query": question_data.question,
                "response": answer[:500] + "..." if len(answer) > 500 else answer,
                "symbol": question_data.symbol,
                "created_at": datetime.utcnow().isoformat()
            }
            
            supabase.table('chat_history').insert(chat_data).execute()
        except Exception as chat_error:
            print(f"Chat history save error: {str(chat_error)}")
        
        return {
            "question": question_data.question,
            "answer": answer,
            "symbol": question_data.symbol
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Stock Research Analyst API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check API key configuration
    api_status = {
        "gemini_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key"),
        "weather_configured": bool(WEATHER_API_KEY and WEATHER_API_KEY != "your_weather_api_key"),
        "tavily_configured": bool(TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key"),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY and 
                                  SUPABASE_URL != "your_supabase_url" and 
                                  SUPABASE_KEY != "your_supabase_key")
    }
    
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "api_configuration": api_status
    }

@app.get("/test-general")
async def test_general_finance():
    """Test general finance functionality"""
    try:
        test_question = "What is the current market outlook?"
        result = await answer_general_finance_question(test_question)
        return {
            "status": "success",
            "test_question": test_question,
            "response_length": len(result) if result else 0,
            "response_preview": result[:200] + "..." if result and len(result) > 200 else result
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "test_question": "What is the current market outlook?"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)