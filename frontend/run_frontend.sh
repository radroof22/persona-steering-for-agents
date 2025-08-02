#!/bin/bash

# Frontend startup script for Clustered Prompt Personalization Demo

echo "🧠 Starting Clustered Prompt Personalization Frontend..."
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the frontend directory"
    echo "Usage: cd frontend && ./run_frontend.sh"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if conda environment is activated
echo "📦 Checking conda environment..."
if [[ "$CONDA_DEFAULT_ENV" != "sentosa" ]]; then
    echo "⚠️  Warning: sentosa conda environment is not activated."
    echo "   Please run: conda activate sentosa"
    echo "   Then run this script again."
    exit 1
fi

# Check if dependencies are available
echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit, httpx, pandas" 2>/dev/null; then
    echo "❌ Error: Required dependencies are missing."
    echo "   Please ensure the sentosa environment is properly installed:"
    echo "   conda env update -f environment.yml"
    exit 1
fi

# Check API connection
echo "🔗 Checking API connection..."
python3 -c "
import asyncio
import sys
sys.path.append('.')
from api_client import APIClient
from config import API_URL

async def check_api():
    client = APIClient()
    try:
        result = await client.health_check()
        print(f'✅ API is accessible at {API_URL}')
        return True
    except Exception as e:
        print(f'❌ Cannot connect to API at {API_URL}')
        print(f'   Error: {e}')
        print('   Please ensure the backend is running:')
        print('   cd ../app && uvicorn main:app --reload --host 0.0.0.0 --port 8000')
        return False

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
success = loop.run_until_complete(check_api())
loop.close()
exit(0 if success else 1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Warning: API is not accessible. The frontend will show an error message."
    echo "   You can still test the UI, but processing will fail."
    echo ""
fi

# Start Streamlit
echo "🚀 Starting Streamlit application..."
echo "   Frontend will be available at: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run main.py --server.port 8501 --server.address 0.0.0.0 