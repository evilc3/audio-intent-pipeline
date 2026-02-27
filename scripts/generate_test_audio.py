import os
import asyncio
from pathlib import Path
import sys

# Add the project root to sys.path to import app.config
sys.path.append(str(Path(__file__).parent.parent))

from deepgram import DeepgramClient
from pydub import AudioSegment
from app.config import settings
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path("tests/test_audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Deepgram TTS Model
TTS_MODEL = "aura-helios-en"

INTENT_PHRASES = {
    "account_support": "I'm having trouble logging into my account, can you help?",
    "billing": "I have a question about my recent invoice and the charges on my credit card.",
    "technical_support": "The application is crashing every time I try to upload a file.",
    "general_inquiry": "What are your business hours and where are you located?",
    "out_of_scope": "I'd like to order a large pepperoni pizza for delivery.",
    "unclear": "Uh... I... well... maybe... I don't know.",
}

async def generate_audio(text: str, filename_stem: str, format: str = "wav"):
    """
    Generate audio using Deepgram TTS and save it.
    """
    api_key = os.getenv("DEEPGRAM_API_KEY") or settings.deepgram_api_key
    if not api_key:
        print("Error: DEEPGRAM_API_KEY not found in .env or settings.")
        return False


    client = DeepgramClient(api_key=api_key)
    
    try:
        if format == "wav":
            response = client.speak.v1.audio.generate(
                text=text,
                model=TTS_MODEL,
                encoding="linear16",
                container="wav",
                sample_rate=8000, ###for phone calls.
            )

        else:
            response = client.speak.v1.audio.generate(
                text=text,
                model=TTS_MODEL,
                encoding="mp3",
            )
                
        output_path = f"{filename_stem}.{format}"

        # Save the audio file
        with open(output_path, "wb") as audio_file:
            audio_file.write(b"".join(response))

        print(f"Generated {output_path}")
        return True
    except Exception as e:
        print(f"Failed to generate {filename_stem}.{format}: {e}")
        exit()
        return False
    

async def main():
    # 1. Generate intent samples
    print("Generating intent samples...")
    for intent in settings.supported_intents:
        phrase = INTENT_PHRASES.get(intent, f"This is a sample for {intent} intent.")
        await generate_audio(phrase, str(OUTPUT_DIR / f"intent_{intent}"), format="wav")
        await generate_audio(phrase, str(OUTPUT_DIR / f"intent_{intent}"), format="mp3")

    # 2. Generate "too long" file (> 300s)
    print("Generating 'too long' audio...")
    phrase = "This is a sample audio segment that we will loop to create a very long file."
    temp_wav = OUTPUT_DIR / "temp_loop.wav"
    if await generate_audio(phrase, str(OUTPUT_DIR / "temp_loop"), format="wav"):
        audio = AudioSegment.from_wav(temp_wav)
        # settings.max_audio_duration_seconds is 300
        target_duration_ms = (settings.max_audio_duration_seconds + 5) * 1000
        loops = int(target_duration_ms / len(audio)) + 1
        long_audio = audio * loops
        long_audio.export(OUTPUT_DIR / "too_long.wav", format="wav")
        os.remove(temp_wav)
        print(f"Generated too_long.wav (duration: {len(long_audio)/1000}s)")

    # 3. Generate "too short" file (< 1s)
    print("Generating 'too short' audio...")
    if await generate_audio("Hi", str(OUTPUT_DIR / "temp_short"), format="wav"):
        audio = AudioSegment.from_wav(OUTPUT_DIR / "temp_short.wav")
        # settings.min_audio_duration_seconds is 1
        short_audio = audio[:500]  # 0.5 seconds
        short_audio.export(OUTPUT_DIR / "too_short.wav", format="wav")
        os.remove(OUTPUT_DIR / "temp_short.wav")
        print(f"Generated too_short.wav (duration: {len(short_audio)/1000}s)")

    # 4. Generate "no transcript" file (silence within time limit)
    print("Generating 'no transcript' (silent) audio...")
    # 5 seconds of silence
    silence = AudioSegment.silent(duration=5000)
    silence.export(OUTPUT_DIR / "no_transcript.wav", format="wav")
    print("Generated no_transcript.wav (duration: 5.0s)")

    print("\nAll test audio files generated in tests/test_audio/")

if __name__ == "__main__":
    asyncio.run(main())
