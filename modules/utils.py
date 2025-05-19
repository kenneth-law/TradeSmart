"""
Utility Module

This module contains utility functions used across the stock analysis application.
"""

import logging

# Default message handler
message_handler = print

def set_message_handler(handler):
    """
    Sets the global message handler for the application.

    This function allows assigning a custom message processing handler to the
    global `message_handler` variable. The provided handler should be a callable
    function or object that can process incoming messages.

    Parameters:
        handler: Callable
            A callable object or function that will handle the processing of
            messages. The exact behavior of the handler depends on the provided
            implementation.
    """
    global message_handler
    message_handler = handler

def log_message(message):
    """
    Logs a message using the current message handler.
    
    Parameters:
        message (str): The message to log
    """
    message_handler(message)