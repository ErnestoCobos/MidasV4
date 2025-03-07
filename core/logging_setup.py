import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Custom log levels
TRADE = 25  # Between INFO and WARNING

# Set custom log level name
logging.addLevelName(TRADE, "TRADE")

# Debug color codes for terminal
COLORS = {
    'RESET': '\033[0m',
    'RED': '\033[31m',
    'GREEN': '\033[32m',
    'YELLOW': '\033[33m',
    'BLUE': '\033[34m',
    'MAGENTA': '\033[35m',
    'CYAN': '\033[36m',
    'WHITE': '\033[37m',
    'BOLD': '\033[1m'
}

# Map log levels to colors
LEVEL_COLORS = {
    'DEBUG': COLORS['BLUE'],
    'INFO': COLORS['GREEN'],
    'TRADE': COLORS['CYAN'],
    'WARNING': COLORS['YELLOW'],
    'ERROR': COLORS['RED'],
    'CRITICAL': COLORS['BOLD'] + COLORS['RED']
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages in console"""
    
    def format(self, record):
        # Save original levelname and message
        orig_levelname = record.levelname
        orig_message = record.getMessage()
        
        # Add color to levelname based on level
        if record.levelname in LEVEL_COLORS:
            record.levelname = f"{LEVEL_COLORS[record.levelname]}{record.levelname}{COLORS['RESET']}"
            
        # Add prefix tags to message based on content
        message = orig_message
        
        # Check for common message patterns and add tags
        prefixes = {
            "trade": "[TRADE EXECUTION]",
            "position": "[POSITION]",
            "signal": "[STRATEGY SIGNAL]",
            "commission": "[COMMISSION]",
            "closed": "[TRADE CLOSED]",
            "stop loss": "[STOP LOSS]",
            "take profit": "[TAKE PROFIT]",
            "trailing stop": "[TRAILING STOP]",
            "performance": "[PERFORMANCE]",
            "balance": "[BALANCE]"
        }
        
        # Look for message patterns to tag appropriately
        tagged = False
        lower_msg = orig_message.lower()
        
        for key, prefix in prefixes.items():
            if key in lower_msg and not tagged:
                # Add colored tag prefix to message
                color = COLORS['CYAN'] if key != "stop loss" else COLORS['RED']
                color = COLORS['GREEN'] if key == "take profit" else color
                record.msg = f"{color}{prefix}{COLORS['RESET']} {record.msg}"
                tagged = True
                break
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname and message
        record.levelname = orig_levelname
        record.msg = orig_message
        
        return result

def is_debug_mode():
    """Check if debug mode is enabled through environment variable"""
    return os.environ.get("MIDAS_DEBUG", "0").lower() in ("1", "true", "yes", "y")

def get_detailed_formatter():
    """
    Returns the detailed formatter for debugging
    """
    return logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
    )

def get_standard_formatter():
    """
    Returns the standard formatter for normal operation
    """
    return logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_colored_formatter(detailed=False):
    """
    Returns a colored formatter for console output
    """
    if detailed:
        return ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
        )
    else:
        return ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

def trade(self, message, *args, **kwargs):
    """
    Custom log method for trade-specific logs
    """
    self.log(TRADE, message, *args, **kwargs)

def add_custom_log_methods(logger):
    """
    Add custom log methods to the logger
    """
    # Add trade method
    logger.trade = trade.__get__(logger)
    
    # Add convenience methods
    logger.success = lambda msg, *args, **kwargs: logger.info(f"✅ {msg}", *args, **kwargs)
    logger.fail = lambda msg, *args, **kwargs: logger.error(f"❌ {msg}", *args, **kwargs)
    
    return logger

def setup_logging(
    name, 
    level=None, 
    log_to_file=True, 
    log_dir='logs',
    component=None,
    detailed_format=False
):
    """
    Set up logging configuration with enhanced debugging capabilities
    
    Args:
        name: Logger name
        level: Logging level (if None, will use DEBUG in debug mode, INFO otherwise)
        log_to_file: Whether to log to file
        log_dir: Directory to store log files
        component: Component name for filtering (optional)
        detailed_format: Use detailed format with line numbers (default: based on debug mode)
    
    Returns:
        Logger instance
    """
    # Determine the log level based on debug mode
    debug_mode = is_debug_mode()
    
    # Set default level based on debug mode if not specified
    if level is None:
        level = logging.DEBUG if debug_mode else logging.INFO
    
    # Set detailed_format based on debug mode if not specified
    if detailed_format is None:
        detailed_format = debug_mode
    
    # Create logger
    if component:
        logger_name = f"{name}.{component}"
    else:
        logger_name = name
        
    logger = logging.getLogger(logger_name)
    
    # Remove existing handlers to prevent duplicate logs
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Set level
    logger.setLevel(level)
    
    # Choose formatter based on debug mode
    if detailed_format:
        formatter = get_detailed_formatter()
        colored_formatter = get_colored_formatter(detailed=True)
    else:
        formatter = get_standard_formatter()
        colored_formatter = get_colored_formatter(detailed=False)
    
    # Create console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(colored_formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create log file path with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        
        # Add component to filename if provided
        if component:
            log_file = os.path.join(log_dir, f'{name}_{component}_{timestamp}.log')
        else:
            log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        # Create debug file handler if in debug mode
        if debug_mode:
            debug_file = os.path.join(log_dir, f'{name}_debug_{timestamp}.log')
            debug_handler = RotatingFileHandler(
                debug_file,
                maxBytes=20 * 1024 * 1024,  # 20 MB
                backupCount=10
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(get_detailed_formatter())
            logger.addHandler(debug_handler)
        
        # Create standard file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)  # Always keep INFO level for standard log file
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    # Add custom log methods
    logger = add_custom_log_methods(logger)
    
    # Log startup information for debugging
    if debug_mode:
        logger.debug(f"Logger '{logger_name}' initialized in DEBUG mode")
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Platform: {sys.platform}")
        logger.debug(f"Current working directory: {os.getcwd()}")
    
    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds transaction ID to logs
    """
    def process(self, msg, kwargs):
        if 'txid' in self.extra:
            return f"[TXID:{self.extra['txid']}] {msg}", kwargs
        return msg, kwargs

def get_transaction_logger(logger, txid):
    """
    Creates a transaction-specific logger adapter
    
    Args:
        logger: Base logger instance
        txid: Transaction ID to add to logs
    
    Returns:
        LoggerAdapter instance
    """
    return LoggerAdapter(logger, {'txid': txid})

def log_exception(logger, e, context=None):
    """
    Logs an exception with full stack trace in debug mode
    
    Args:
        logger: Logger instance
        e: Exception to log
        context: Additional context for the error
    """
    if context:
        error_msg = f"{context}: {str(e)}"
    else:
        error_msg = str(e)
    
    if is_debug_mode():
        logger.error(f"Exception: {error_msg}\n{traceback.format_exc()}")
    else:
        logger.error(error_msg)

# Global exception hook to log unhandled exceptions
def global_exception_hook(exctype, value, tb):
    """
    Global exception hook to log unhandled exceptions
    """
    logger = logging.getLogger('unhandled_exceptions')
    
    # Set up logger if not already configured
    if not logger.handlers:
        setup_logging('unhandled_exceptions', level=logging.ERROR)
    
    # Log the exception
    logger.critical(f"Unhandled {exctype.__name__}: {value}")
    
    if is_debug_mode():
        logger.critical(''.join(traceback.format_tb(tb)))
    
    # Call the original exception hook
    sys.__excepthook__(exctype, value, tb)

# Install the global exception hook
sys.excepthook = global_exception_hook