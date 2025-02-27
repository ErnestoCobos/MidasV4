class CredentialsError(Exception):
    """Raised when there is an issue with API credentials"""
    pass

class APIError(Exception):
    """Raised when there is an error with the exchange API"""
    pass

class StrategyError(Exception):
    """Raised when there is an error in the trading strategy"""
    pass

class OrderError(Exception):
    """Raised when there is an error creating or managing orders"""
    pass

class RiskManagementError(Exception):
    """Raised when risk management constraints are violated"""
    pass