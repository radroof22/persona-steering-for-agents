#!/bin/bash

# Run tests for the clusterer service
echo "Running clusterer tests..."
python -m pytest tests/test_clusterer.py -v

echo ""
echo "Running rewriter tests..."
python -m pytest tests/test_rewriter.py -v

echo ""
echo "Running main API tests..."
python -m pytest tests/test_main.py -v

echo ""
echo "Running all tests..."
python -m pytest tests/ -v

# Run with coverage if available
if command -v coverage &> /dev/null; then
    echo "Running tests with coverage..."
    coverage run -m pytest tests/
    coverage report
else
    echo "Coverage not available. Install with: pip install coverage"
fi 