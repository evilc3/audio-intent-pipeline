VERSIONS: dict[str, dict] = {
    "v1.0": {
        "system": """\
You are an intent classification system for a voice agent.
Your job is to analyse a user transcript and return a structured response.

Rules:
- Supported intents: account_support, billing, technical_support, 
  general_inquiry, out_of_scope, unclear
- Use "unclear" if you cannot determine intent with reasonable certainty
- Use "out_of_scope" if the request is clear but outside system capabilities
- Confidence must be a float between 0 and 1
- Always suggest a concrete next action based on the intent
- Input may be English or Italian — handle both
- If STT confidence is low, factor this into your own confidence score
""",
        "user_template": (
            "Transcript: {transcript}\n"
            "Language: {language}\n"
            "STT Confidence: {stt_confidence}\n\n"
            "Classify the intent and suggest an action."
        ),
    },
    "v1.1": {
        # future version goes here
        "system": "...",
        "user_template": "...",
    }
}

def get_prompt(version: str) -> dict:
    if version not in VERSIONS:
        raise ValueError(f"Prompt version '{version}' not found. Available: {list(VERSIONS.keys())}")
    return VERSIONS[version]

def build_user_prompt(version: str, transcript: str, language: str, stt_confidence: float) -> str:
    prompt = get_prompt(version)
    return prompt["user_template"].format(
        transcript=transcript,
        language=language,
        stt_confidence=f"{stt_confidence:.2f}" if stt_confidence is not None else "N/A"
    )

def build_system_prompt(version:str) -> str:
    prompt = get_prompt(version)
    return prompt["system"]