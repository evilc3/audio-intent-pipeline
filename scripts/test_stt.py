import os
from pathlib import Path
from deepgram import DeepgramClient
from app.config import settings
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

INPUT_DIR = Path("tests/test_audio")
OUTPUT_DIR = Path("tests/test_stt_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

##getting the API key.
API_KEY = os.getenv("DEEPGRAM_API_KEY") or settings.deepgram_api_key

if not API_KEY:
    print("Deepgram API key not found.")
    exit()

CLIENT = DeepgramClient(api_key=API_KEY)

def transcribe_audio(audio_file: Path):

    print(f"Testing File: {audio_file}")

    try:

        with open(audio_file, "rb") as file:
            response = CLIENT.listen.v1.media.transcribe_file(
                request=file.read(),
                model=settings.deepgram_model,
                smart_format=True,
                language="en",
                punctuate=True,
                keyterm=["account", "help"], ##just for demo.
            )
        
            pprint(response.dict())
    
    except Exception as e:
        print(f"Failed to transcribe {audio_file}: {e}")
        return None

def main():
    # transcribe_audio(INPUT_DIR / "intent_account_support.wav") #working
    transcribe_audio(INPUT_DIR / "intent_account_support.mp3")

if __name__ == "__main__":
    main()