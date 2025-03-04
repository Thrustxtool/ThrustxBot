# ========================
# Trading Parameters
# ========================
TRADING_PARAMS = {
    'symbols': ['SPY', 'QQQ', 'IWM'],  # Default trading universe
    'trading_hours': {  # Market hours in UTC
        'start': '14:30',  # 9:30 AM ET
        'end': '21:00'     # 4:00 PM ET
    },
    'max_drawdown': 0.2,  # Max portfolio drawdown
    'daily_loss_limit': 2.0,  # Max daily loss percentage
    'position_size': 5.0  # Max position size percentage
}

# ========================
# Feature Configuration
# ========================
FEATURE_CONFIG = {
    'technical_indicators': {
        'rsi': {'window': 14},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'atr': {'window': 14},
        'obv': {}
    },
    'sentiment_sources': ['reddit', 'twitter', 'news'],
    'data_resolution': '15min'  # 1min, 5min, 15min, 1H
}

# ========================
# Model Configuration
# ========================
MODEL_CONFIG = {
    'lstm': {
        'lookback': 60,
        'epochs': 20,
        'batch_size': 32,
        'dropout_rate': 0.2,
        'hidden_units': 50
    },
    'prophet': {
        'seasonality_mode': 'multiplicative',
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0
    }
}

# ========================
# Risk Management
# ========================
RISK_CONFIG = {
    'volatility_adjustment': 2.0,  # ATR multiplier for stops
    'sector_exposure_limit': 0.3,  # Max exposure per sector
    'portfolio_heat_limit': 0.8,   # Max portfolio utilization
    'stop_loss': {
        'initial': 0.02,  # 2% initial stop
        'trailing': 0.01   # 1% trailing stop
    }
}

# ========================
# Compliance Settings
# ========================
COMPLIANCE_CONFIG = {
    'audit_log_path': 'data/compliance_logs/',
    'trade_reconstruction': True,
    'regulatory_rules': {
        'finra_613': True,  # Large trader reporting
        'mifid_ii': False,   # EU reporting
        'sec_13h': False     # SEC large trader
    }
}

# ========================
# Execution Parameters
# ========================
EXECUTION_CONFIG = {
    'order_types': {
        'market': True,
        'limit': True,
        'twap': True
    },
    'slippage_control': {
        'max_slippage': 0.005,  # 0.5%
        'adaptive': True
    },
    'venue_selection': {
        'dark_pools': 0.4,
        'lit_exchanges': 0.6
    }
}

# ========================
# System Monitoring
# ========================
MONITORING_CONFIG = {
    'resource_limits': {
        'cpu': 85.0,  # Max CPU usage percentage
        'memory': 80.0,  # Max memory usage percentage
        'disk': 90.0  # Max disk usage percentage
    },
    'alert_thresholds': {
        'drawdown': 0.15,
        'daily_loss': 1.5,
        'model_accuracy': 0.55
    }
}