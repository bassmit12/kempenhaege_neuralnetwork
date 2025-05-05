#!/usr/bin/env python
"""
Main entry point for the AI Scheduling Neural Network service.
This script initializes and starts the API server.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'neural_network.log'))
    ]
)
logger = logging.getLogger('neural_network')

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='AI Scheduling Neural Network Service')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    logger.info(f"Starting AI Scheduling Neural Network Service on {args.host}:{args.port}")
    logger.info(f"Debug mode: {args.debug}")
    
    # Import the Flask app from the API module
    try:
        from api.server import app
        
        # Create required directories
        os.makedirs(os.path.join(os.path.dirname(__file__), 'saved_models'), exist_ok=True)
        
        # Run the Flask app
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()