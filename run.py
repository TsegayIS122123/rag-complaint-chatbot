#!/usr/bin/env python
"""
Fast startup script for CrediTrust Chatbot
Disables warnings and optimizes loading
"""

import os
import sys

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Import app
import app

if __name__ == "__main__":
    app.demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=False,
        quiet=True
    )