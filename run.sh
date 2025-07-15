#!/bin/bash

# Accident Severity Prediction ML Pipeline - Run Script

echo "🚀 Starting Accident Severity Prediction ML Pipeline"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if data files exist
if [ ! -f "data/california_data.csv" ]; then
    echo "⚠️  Warning: california_data.csv not found in data/ directory"
    echo "   Please ensure your data files are in the data/ directory"
fi

if [ ! -f "data/florida_data.csv" ]; then
    echo "⚠️  Warning: florida_data.csv not found in data/ directory"
    echo "   Please ensure your data files are in the data/ directory"
fi

# Run tests (optional)
echo "🧪 Running pipeline tests..."
python test_pipeline.py

echo ""
echo "🌐 Starting Flask application..."
echo "   Open your browser and go to: http://localhost:5001"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
python app.py