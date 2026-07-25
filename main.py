from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv
load_dotenv()
nvidia_api_key = os.environ.get("nvidia_api_key")
v4_flash = ChatNVIDIA(
    model="deepseek-ai/deepseek-v4-flash",
    nvidia_api_key=nvidia_api_key,)
diffusiongemma = ChatNVIDIA(
  model="google/diffusiongemma-26b-a4b-it",
  nvidia_api_key=nvidia_api_key,
)
response = v4_flash.invoke("Hello, how are you?")
print(response.content)
