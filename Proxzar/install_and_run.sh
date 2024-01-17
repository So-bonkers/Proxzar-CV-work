#!/bin/bash

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Run vit.py
echo "Running vit.py..."
python vit.py

# Run trainer.py
echo "Running trainer.py..."
python trainer.py
