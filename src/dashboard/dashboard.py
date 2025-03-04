import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import psutil
from typing import Dict, Any
from collections import defaultdict

class TradingDashboard:
    def __init__(self, bot):
        self.bot = bot
        st.set_page_config(layout="wide")
        self.initialize_styles()
        
    def initialize_styles(self):
        st.markdown("""
        <style>
            .metric-box {
                padding: 15px;
                border-radius: 10px;
                background-color: #f0f2f6;
                margin: 10px 0;
            }
            .plot-container {
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #e6e6e6;
                border-radius: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

    def render(self):
        """Main dashboard rendering function"""
        st.title("AI Trading Bot Dashboard")
        
        try:
            with st.spinner("Loading dashboard components..."):
                self._render_system_monitor()
                self._render_market_overview()
                self._render_portfolio()
                self._render_execution_history()
                self._render_model_performance()
                
        except Exception as e:
            st.error(f"Dashboard rendering failed: {str(e)}")

    def _render_system_monitor(self):
        """System resource monitoring section"""
        st.subheader("System Resources")
        col1, col2, col3 = st.columns(3)
        
        try:
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
                st.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Disk Usage", f"{psutil.disk_usage('/').percent}%")
                st.metric("Network IO", 
                         f"{psutil.net_io_counters().bytes_sent/1e6:.2f} MB Sent")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                if hasattr(self.bot, 'start_time'):
                    uptime = str(datetime.now() - self.bot.start_time).split('.')[0]
                    st.metric("Bot Uptime", uptime)
                if hasattr(self.bot, 'order_history'):
                    st.metric("Trade Count", len(self.bot.order_history))
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"System monitoring error: {str(e)}")

    def _render_market_overview(self):
        """Market data visualization section"""
        st.subheader("Market Overview")
        try:
            if not hasattr(self.bot, 'market_data') or self.bot.market_data.empty:
                st.warning("⌛ Waiting for initial market data...")
                return

            df = self.bot.market_data
            required_columns = ['close', 'rsi', 'macd']
            
            if not all(col in df.columns for col in required_columns):
                st.warning("⏳ Initializing technical indicators...")
                return

            with st.expander("Price Movement", expanded=True):
                fig = px.line(df, x=df.index, y='close', 
                            title="Price Movement", 
                            labels={'close': 'Price', 'index': 'Time'})
                st.plotly_chart(fig, use_container_width=True)

            cols = st.columns(2)
            with cols[0]:
                with st.expander("RSI Analysis", expanded=True):
                    fig = px.line(df, x=df.index, y='rsi', 
                                title="Relative Strength Index",
                                labels={'rsi': 'RSI', 'index': 'Time'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with cols[1]:
                with st.expander("MACD Analysis", expanded=True):
                    fig = px.line(df, x=df.index, y='macd', 
                                title="Moving Average Convergence Divergence",
                                labels={'macd': 'MACD', 'index': 'Time'})
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"⚠️ Market data error: {str(e)}")

    def _render_portfolio(self):
        """Portfolio management section"""
        st.subheader("Portfolio Management")
        try:
            if not hasattr(self.bot, 'risk') or not hasattr(self.bot.risk, 'portfolio'):
                st.warning("⏳ Portfolio system initializing...")
                return

            portfolio = self.bot.risk.portfolio
            if not portfolio.get('positions'):
                st.info("💵 Current Cash: ${:,.2f}".format(portfolio.get('cash', 0)))
                return

            # Create portfolio breakdown
            positions = []
            for symbol, details in portfolio['positions'].items():
                positions.append({
                    'Asset': symbol,
                    'Value': details['value'],
                    'Quantity': details['quantity'],
                    'Entry Price': details['entry_price'],
                    'Current Price': details.get('current_price', details['entry_price'])
                })

            df = pd.DataFrame(positions)
            total_value = portfolio['cash'] + df['Value'].sum()

            cols = st.columns([2, 3])
            with cols[0]:
                st.markdown("### Portfolio Breakdown")
                st.dataframe(df.style.format({
                    'Value': '${:,.2f}',
                    'Entry Price': '${:,.2f}',
                    'Current Price': '${:,.2f}'
                }))

            with cols[1]:
                st.markdown("### Asset Allocation")
                allocation_df = pd.DataFrame({
                    'Asset': ['Cash'] + df['Asset'].tolist(),
                    'Value': [portfolio['cash']] + df['Value'].tolist()
                })
                fig = px.pie(allocation_df, values='Value', names='Asset', 
                           title="Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**Total Portfolio Value:** ${total_value:,.2f}")

        except Exception as e:
            st.error(f"⚠️ Portfolio error: {str(e)}")

    def _render_execution_history(self):
        """Trade execution history section"""
        st.subheader("Execution History")
        try:
            if not hasattr(self.bot, 'order_history') or len(self.bot.order_history) == 0:
                st.info("📭 No trades executed yet")
                return

            history = pd.DataFrame(self.bot.order_history)
            if history.empty:
                st.warning("📭 Empty trade history")
                return

            with st.expander("Recent Trades", expanded=True):
                st.dataframe(history.style.format({
                    'price': '${:.2f}',
                    'qty': '{:.2f}'
                }))

            st.markdown("### Execution Timeline")
            history['timestamp'] = pd.to_datetime(history['timestamp'])
            fig = px.bar(history, x='timestamp', y='qty', color='side',
                        labels={'qty': 'Quantity', 'timestamp': 'Time'},
                        title="Trade Execution History")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Execution history error: {str(e)}")

    def _render_model_performance(self):
        """Model performance metrics section"""
        st.subheader("Model Performance")
        try:
            if not hasattr(self.bot, 'model_performance'):
                st.warning("⏳ Model metrics collecting...")
                return

            metrics = self.bot.model_performance
            if not metrics.get('predictions'):
                st.info("📊 No performance data collected yet")
                return

            cols = st.columns(2)
            with cols[0]:
                st.markdown("### Predictions vs Actuals")
                plot_data = pd.DataFrame({
                    'Actual': metrics.get('actuals', []),
                    'Predicted': metrics.get('predictions', [])
                })
                if not plot_data.empty:
                    fig = px.line(plot_data, 
                                title="Price Predictions vs Actual Values",
                                labels={'value': 'Price', 'index': 'Sequence'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No prediction data available")

            with cols[1]:
                st.markdown("### Accuracy History")
                if metrics.get('accuracy_history'):
                    accuracy_df = pd.DataFrame({
                        'Accuracy': metrics['accuracy_history'],
                        'Epoch': range(1, len(metrics['accuracy_history']) + 1)
                    })
                    fig = px.line(accuracy_df, x='Epoch', y='Accuracy',
                                title="Model Accuracy Over Time",
                                labels={'Accuracy': 'Accuracy (%)'})
                    fig.update_layout(yaxis_tickformat=".2%")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("⏳ Accuracy metrics being calculated...")

        except Exception as e:
            st.error(f"⚠️ Performance error: {str(e)}")

def main(bot):
    """Dashboard entry point"""
    try:
        dashboard = TradingDashboard(bot)
        dashboard.render()
    except Exception as e:
        st.error(f"🚨 Dashboard failed to initialize: {str(e)}")

if __name__ == "__main__":
    from main import TradingBot
    try:
        bot = TradingBot()
        main(bot)
    except Exception as e:
        st.error(f"🚨 Bot initialization failed: {str(e)}")