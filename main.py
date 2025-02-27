import json
import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import assemblyai as aai
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("minutes_creation.log"),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Keys
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Directory settings
AUDIO_DIR = os.getenv("AUDIO_DIR", "audio_files")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_files")
TRANSCRIPTS_DIR = f"{OUTPUT_DIR}/transcripts"
SUMMARIES_DIR = f"{OUTPUT_DIR}/summaries"

# Supported audio file extensions
SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".flac", ".aac", ".ogg"}

# prompt
initial_template = """以下の会話記録から正式な議事録を作成してください。以下の要素を含めて下さい：
1. 会議日時：[自動挿入]
2. 参加者：[役職・名前]
3. 決定事項（箇条書き）
4. 保留課題
5. 次回予定

会話内容：
{text}

出力形式：
・Markdownを使用
・敬体（です・ます調）で統一
・専門用語は原語維持
・数値データは表形式で整理"""

refine_template = """既存の議事録を以下の新規内容で更新してください：
{existing_answer}

新規会話内容：
{text}

更新ルール：
1. 重複情報は除外
2. 数値データは累積計算
3. 決定事項は時系列順に整理
4. 変更点を[UPDATED]タグで明示"""


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)  # 半角・全角統一
    text = re.sub(r"[\u3000-\u303F]", " ", text)  # 記号除去
    text = re.sub(r"\s+", " ", text)  # 連続空白削除
    return text.strip()


class ProcessedFile(BaseModel):
    """Model for tracking processed files."""

    filename: str
    original_path: str
    transcript_path: Optional[str] = None
    summary_path: Optional[str] = None
    processed_date: datetime = Field(default_factory=datetime.now)
    file_size: int
    duration_seconds: Optional[float] = None
    status: str = "pending"  # pending, transcribed, summarized, error
    error_message: Optional[str] = None


class MinutesCreator:
    """Main class for creating minutes from audio files."""

    def __init__(self):
        """Initialize the MinutesCreator."""
        self._setup_directories()
        self._setup_clients()
        self.processed_files_path = Path(OUTPUT_DIR) / "processed_files.json"
        self.processed_files: Dict[str, ProcessedFile] = self._load_processed_files()

    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [AUDIO_DIR, OUTPUT_DIR, TRANSCRIPTS_DIR, SUMMARIES_DIR]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")

    def _setup_clients(self) -> None:
        """Set up API clients."""
        if not ASSEMBLYAI_API_KEY:
            raise ValueError("ASSEMBLYAI_API_KEY is not set in environment variables")
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment variables")

        # Set up AssemblyAI client
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        logger.info("AssemblyAI client initialized")

        # Set up Anthropic client for LangChain
        self.llm = ChatAnthropic(
            temperature=0,
            model_name="claude-3-7-sonnet-20250219",
            api_key=ANTHROPIC_API_KEY,
        )
        logger.info("Anthropic Claude 3 Sonnet client initialized")

    def _load_processed_files(self) -> Dict[str, ProcessedFile]:
        """Load the list of processed files from JSON."""
        if not self.processed_files_path.exists():
            logger.info("No processed files record found, creating new one")
            return {}

        try:
            with open(self.processed_files_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: ProcessedFile.model_validate(v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Error loading processed files: {e}")
            return {}

    def _save_processed_files(self) -> None:
        """Save the list of processed files to JSON."""
        try:
            with open(self.processed_files_path, "w", encoding="utf-8") as f:
                json.dump(
                    {k: v.model_dump() for k, v in self.processed_files.items()},
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"Saved processed files record to {self.processed_files_path}")
        except Exception as e:
            logger.error(f"Error saving processed files: {e}")

    def scan_audio_files(self) -> List[Path]:
        """Scan the audio directory for files that need processing."""
        audio_dir = Path(AUDIO_DIR)
        all_files = []

        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(audio_dir.glob(f"**/*{ext}"))

        logger.info(f"Found {len(all_files)} audio files in {AUDIO_DIR}")
        return all_files

    def get_unprocessed_files(self, all_files: List[Path]) -> List[Path]:
        """Filter out files that have already been processed."""
        processed_paths = {
            Path(info.original_path)
            for info in self.processed_files.values()
            if info.status in ["transcribed", "summarized"]
        }

        unprocessed = [f for f in all_files if f not in processed_paths]
        logger.info(f"Found {len(unprocessed)} unprocessed audio files")
        return unprocessed

    def transcribe_file(self, file_path: Path) -> Optional[str]:
        """Transcribe an audio file using AssemblyAI with speaker diarization and Japanese language."""
        try:
            logger.info(f"Transcribing file: {file_path}")

            # Create a unique ID for this file based on path and modification time
            file_stat = file_path.stat()
            file_id = f"{file_path}_{file_stat.st_mtime}"

            # Check if we've already started processing this file
            if file_id in self.processed_files:
                logger.info(f"File {file_path} is already in the processing queue")
                return None

            # Create a new record for this file
            self.processed_files[file_id] = ProcessedFile(
                filename=file_path.name,
                original_path=str(file_path),
                file_size=file_stat.st_size,
                status="pending",
            )
            self._save_processed_files()

            # Configure transcription with speaker labels and Japanese language
            config = aai.TranscriptionConfig(speaker_labels=True, language_code="ja")

            # Start transcription
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(str(file_path), config)

            if not transcript.text:
                raise ValueError("Transcription returned empty text")

            # Save the transcript
            transcript_path = Path(TRANSCRIPTS_DIR) / f"{file_path.stem}.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript.text)

            # Save the transcript with speaker diarization
            diarization_path = Path(TRANSCRIPTS_DIR) / f"{file_path.stem}_diarization.txt"
            speaker_transcript = []
            for utterance in transcript.utterances:
                speaker_transcript.append(f"Speaker {utterance.speaker}: {utterance.text}")

            with open(diarization_path, "w", encoding="utf-8") as f:
                f.write("\n".join(speaker_transcript))

            # Update the record
            self.processed_files[file_id].transcript_path = str(
                diarization_path
            )  # Use the diarization transcript for summarization
            self.processed_files[file_id].duration_seconds = transcript.audio_duration
            self.processed_files[file_id].status = "transcribed"
            self._save_processed_files()

            logger.info(f"Transcription completed and saved to {transcript_path}")
            logger.info(f"Speaker diarization transcript saved to {diarization_path}")
            return str(diarization_path)  # Return the diarization transcript path for summarization

        except Exception as e:
            logger.error(f"Error transcribing {file_path}: {e}")
            if file_id in self.processed_files:
                self.processed_files[file_id].status = "error"
                self.processed_files[file_id].error_message = str(e)
                self._save_processed_files()
            return None

    def summarize_transcript(self, transcript_path: str) -> Optional[str]:
        """Summarize a transcript using LangChain."""
        try:
            logger.info(f"Summarizing transcript: {transcript_path}")

            # Find the corresponding file record
            file_id = next((k for k, v in self.processed_files.items() if v.transcript_path == transcript_path), None)

            if not file_id:
                logger.warning(f"No record found for transcript {transcript_path}")
                return None

            # Read the transcript
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()

            # Normalize the text
            transcript_text = normalize_text(transcript_text)

            # Split the text into chunks for processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,  # 日本語の平均文長を考慮
                chunk_overlap=500,  # 文脈継続のため増加
                separators=["\n\n", "。", "、", " "],  # 日本語の句読点を優先
                length_function=len,
                is_separator_regex=False,
            )

            text_splitter = SpacyTextSplitter(
                chunk_size=3000, pipeline="ja_core_news_sm", separators=["。", "？", "！", "\n\n"]
            )

            docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(transcript_text)]

            # Load the summarization chain
            chain = load_summarize_chain(
                self.llm,
                chain_type="refine",
                verbose=True,
                question_prompt=initial_template,
                refine_prompt=refine_template,
            )

            # Generate the summary
            summary = chain.run(docs)

            # Save the summary
            transcript_path_obj = Path(transcript_path)
            summary_path = Path(SUMMARIES_DIR) / f"{transcript_path_obj.stem}_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)

            # Update the record
            self.processed_files[file_id].summary_path = str(summary_path)
            self.processed_files[file_id].status = "summarized"
            self._save_processed_files()

            logger.info(f"Summary completed and saved to {summary_path}")
            return str(summary_path)

        except Exception as e:
            logger.error(f"Error summarizing {transcript_path}: {e}")
            if file_id:
                self.processed_files[file_id].status = "error"
                self.processed_files[file_id].error_message = str(e)
                self._save_processed_files()
            return None

    def process_all_files(self) -> None:
        """Process all unprocessed audio files."""
        # Scan for audio files
        all_files = self.scan_audio_files()
        unprocessed_files = self.get_unprocessed_files(all_files)

        if not unprocessed_files:
            logger.info("No new files to process")
            return

        # Process each file
        for file_path in unprocessed_files:
            logger.info(f"Processing file: {file_path}")

            # Transcribe
            transcript_path = self.transcribe_file(file_path)
            if not transcript_path:
                logger.warning(f"Skipping summarization for {file_path} due to transcription failure")
                continue

            # Summarize
            summary_path = self.summarize_transcript(transcript_path)
            if not summary_path:
                logger.warning(f"Summarization failed for {transcript_path}")
                continue

            logger.info(f"Successfully processed {file_path}")

        logger.info("Finished processing all files")


def main():
    """Main entry point for the application."""
    try:
        logger.info("Starting minutes creation tool")
        creator = MinutesCreator()
        creator.process_all_files()
        logger.info("Minutes creation completed successfully")
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)


if __name__ == "__main__":
    main()
