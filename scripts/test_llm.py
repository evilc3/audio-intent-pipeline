import os
import asyncio

from openai import AsyncAzureOpenAI
from app.config import settings

api_key = os.getenv("AZURE_OPENAI_KEY") or settings.azure_openai_key
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or settings.azure_openai_endpoint
primary_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_PRIMARY") or settings.azure_openai_deployment_primary
fallback_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_FALLBACK") or settings.azure_openai_deployment_fallback

print(api_key)
print(azure_endpoint)
print(primary_deployment)
print(fallback_deployment)

client = AsyncAzureOpenAI(
    api_key=api_key ,
    azure_endpoint=azure_endpoint,
    api_version="2024-06-01",
)

async def test_primary():
    response = await client.chat.completions.create(
        model=primary_deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )
    print(response.choices[0].message.content)

async def test_fallback():
    response = await client.chat.completions.create(
        model=fallback_deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )
    print(response.choices[0].message.content)

async def main():
    await test_primary()
    await test_fallback()

if __name__ == "__main__":
    asyncio.run(main())