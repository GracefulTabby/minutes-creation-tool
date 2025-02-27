#!/usr/bin/env python3
"""
Test script for the Minutes Creation Tool.

This script demonstrates how to use the MinutesCreator class directly
with a specific audio file, rather than scanning a directory.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from main import MinutesCreator


def test_with_file(audio_file_path: str) -> None:
    """
    Test the MinutesCreator with a specific audio file.

    Args:
        audio_file_path: Path to the audio file to process
    """
    # Check if file exists
    file_path = Path(audio_file_path)
    if not file_path.exists():
        print(f"Error: File {audio_file_path} does not exist")
        sys.exit(1)

    # Initialize the MinutesCreator
    creator = MinutesCreator()

    # Process the file
    print(f"Processing file: {file_path}")

    # Force reprocessing by removing any existing records for this file
    file_stat = file_path.stat()
    file_id = f"{file_path}_{file_stat.st_mtime}"
    if file_id in creator.processed_files:
        print(f"Removing existing record for {file_path} to force reprocessing")
        del creator.processed_files[file_id]
        creator._save_processed_files()

    # Transcribe
    transcript_path = creator.transcribe_file(file_path)
    if not transcript_path:
        print(f"Error: Transcription failed for {file_path}")
        sys.exit(1)

    print(f"Transcription successful. Transcript saved to: {transcript_path}")

    # Summarize
    summary_path = creator.summarize_transcript(transcript_path)
    if not summary_path:
        print(f"Error: Summarization failed for {transcript_path}")
        sys.exit(1)

    print(f"Summarization successful. Summary saved to: {summary_path}")
    print("\nProcessing complete!")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Check if API keys are set
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        print("Error: ASSEMBLYAI_API_KEY is not set in .env file")
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set in .env file")
        sys.exit(1)

    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_tool.py <path_to_audio_file>")
        sys.exit(1)

    # Run the test
    test_with_file(sys.argv[1])
