#!/bin/bash
# Setup script for Minutes Creation Tool

# Check if Python 3.13+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
if [[ -z "$python_version" ]]; then
  echo "Error: Python 3 is not installed"
  exit 1
fi

major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

if [[ $major -lt 3 || ($major -eq 3 && $minor -lt 13) ]]; then
  echo "Error: Python 3.13 or higher is required (found $python_version)"
  echo "Please upgrade your Python installation"
  exit 1
fi

echo "Python $python_version found"

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
  echo "Creating .env file from template..."
  cp .env.example .env
  echo "Please edit .env file to add your API keys"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p audio_files output_files/transcripts output_files/summaries

echo ""
echo "Setup complete!"
echo ""
echo "To use the tool:"
echo "1. Edit .env file to add your API keys"
echo "2. Place audio files in the 'audio_files' directory"
echo "3. Run 'python main.py' to process all files"
echo "   or 'python test_tool.py path/to/audio/file.mp3' for a single file"
echo ""
echo "Transcripts and summaries will be saved in the 'output_files' directory"
