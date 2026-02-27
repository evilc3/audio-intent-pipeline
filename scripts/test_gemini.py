import os
import json
from google import genai
from pydantic import BaseModel, Field

# Load environment variable via python-dotenv if available, or assume it's set
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class IntentResponse(BaseModel):
    intent: str = Field(description="The primary intent identified in the transcript.")
    confidence: float = Field(description="Confidence score for the intent classification, from 0.0 to 1.0.")
    action: str = Field(description="Recommended next action to take based on the intent.")
    notes: str | None = Field(default=None, description="Any additional context or reasoning.")

def list_available_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Please set your GEMINI_API_KEY in the .env file or environment.")
        return

    client = genai.Client(api_key=api_key)
    print("--- Available Gemini Models ---")
    try:
        # Note: Depending on the SDK version, this might be client.models.list() or similar
        # Fallback to the google.generativeai if needed, but google-genai should have it.
        # It's usually client.models.list() or client.models.list_models()
        # Let's try client.models.list() 
        # (If this fails, we can catch and try something else)
        models = client.models.list()
        for m in models:
             print(f"- {m.name}")
             # We can print supported generation methods if available
             if hasattr(m, 'supported_generation_methods'):
                 print(f"  Methods: {m.supported_generation_methods}")
    except AttributeError:
        # Fallback if the SDK is older
        pass
    except Exception as e:
         print(f"Error listing models: {e}")
    print("-------------------------------\n")

def test_gemini_structured_output():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Please set your GEMINI_API_KEY in the .env file or environment.")
        return

    # Initialize client
    client = genai.Client(api_key=api_key)
    
    prompt = """
    Transcript: Hi, yeah, I'm trying to figure out why I got charged $40 twice this month. I think there's a mistake on my bill.
    Language: en-US
    STT Confidence: 0.98

    Classify the intent and suggest an action. Supported intents: account_support, billing, technical_support, general_inquiry, out_of_scope, unclear.
    """
    
    # You might need to change this model after seeing the list output
    # model_name = "gemini-3-flash-preview"
    model_name = "gemini-2.5-flash"
    print(f"Calling {model_name} with structured output...")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": IntentResponse,
                "temperature": 0.0
            },
        )
        
        # In the new google-genai SDK, Pydantic objects are returned natively if response_schema is a Pydantic model
        if response.parsed is not None:
            parsed = response.parsed
            print("\nSuccessfully parsed Pydantic object:")
            print(f"Intent: {parsed.intent}")
            print(f"Confidence: {parsed.confidence}")
            print(f"Action: {parsed.action}")
            print(f"Notes: {parsed.notes}")
        else:
            print("\nRaw JSON Text:")
            print(response.text)
            parsed = IntentResponse.model_validate_json(response.text)
            print("\nSuccessfully validated JSON into IntentResponse")
            
    except Exception as e:
        print(f"Error calling model: {e}")

if __name__ == "__main__":
    # list_available_models()
    test_gemini_structured_output()
