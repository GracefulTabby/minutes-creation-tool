# Minutes Creation Tool (AI議事録作成ツール)

A tool for automatically transcribing audio files and creating summarized meeting minutes using AI.

## Features

- Automatically scans a directory for audio files
- Transcribes audio files using AssemblyAI
- Summarizes transcriptions using LangChain and Anthropic's Claude 3.7
- Tracks processed files to avoid duplicate processing
- Supports multiple audio formats (.mp3, .mp4, .wav, .m4a, .flac, .aac, .ogg)

## Prerequisites

- Python 3.13 or higher
- AssemblyAI API key ([How to get one](get_api_keys.md#assemblyai-api-key))
- Anthropic API key ([How to get one](get_api_keys.md#anthropic-api-key))

See [get_api_keys.md](get_api_keys.md) for detailed instructions on obtaining API keys.

## Installation

### Automatic Setup (Recommended)

Run the setup script, which will create a virtual environment, install dependencies, and set up the necessary directories:

```
./setup.sh
```

After running the setup script, edit the `.env` file to add your API keys.

### Manual Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/minutes-creation-tool.git
   cd minutes-creation-tool
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -e .
   ```

4. Create a `.env` file with your API keys (copy from `.env.example`):
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file to add your API keys.

5. Create necessary directories:
   ```
   mkdir -p audio_files output_files/transcripts output_files/summaries
   ```

## Usage

### Batch Processing

1. Place your audio files in the `audio_files` directory (or configure a custom directory in `.env`)

2. Run the tool:
   ```
   python main.py
   ```

3. Check the results in the `output_files` directory:
   - Transcripts will be in `output_files/transcripts`
   - Summaries will be in `output_files/summaries`

### Processing a Single File

You can also process a single audio file directly using the test script:

```
python test_tool.py path/to/your/audio/file.mp3
```

This is useful for testing or when you only need to process one file.

## Configuration

You can configure the following settings in your `.env` file:

- `ASSEMBLYAI_API_KEY`: Your AssemblyAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `AUDIO_DIR`: Directory to scan for audio files (default: "audio_files")
- `OUTPUT_DIR`: Directory to save output files (default: "output_files")

## License

[MIT License](LICENSE)
