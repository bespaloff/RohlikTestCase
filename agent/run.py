#!/usr/bin/env python
"""
Main runner script for the Meal Planning Assistant
This script starts both the MCP server and the Gradio app
"""

import asyncio
import subprocess
import sys
import os
import signal
import time
from pathlib import Path

def run_app():
    """Run the main Gradio application"""
    print("🚀 Starting Meal Planning Assistant...")
    print("=" * 50)
    
    # Set environment variables if needed
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("=" * 50)
    
    try:
        print("✅ Starting Gradio interface...")
        print("📍 Access the app at: http://127.0.0.1:7860")
        print("=" * 50)
        print("\n📝 Example requests to try:")
        print("  - 'I need vegetarian meals for 3 days'")
        print("  - 'Plan low-carb meals for the week'")
        print("  - 'Help me with high-protein meals for 5 days'")
        print("\n🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Import and run the app (app.py handles the launch)
        import app
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Meal Planning Assistant...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_app()