#!/bin/bash
# Clara API Backend Runner

echo "=================================="
echo "  Clara API Backend"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run the server
echo ""
echo "Starting server on http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================="

uvicorn main:app --reload --host 0.0.0.0 --port 8000
