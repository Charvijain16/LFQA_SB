import logging
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# from openai import OpenAI

def init(model_name, main_ip, api_key):
    global LLM
    if "llama" in model_name:
        llm = ChatOpenAI(
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            base_url="https://llm.scads.ai/v1",
            openai_api_key=api_key,
            temperature=0.1,
            timeout=300,
        )
    elif "gpt" in model_name:
        llm = ChatOpenAI(
            model=model_name, 
            api_key=api_key,)
            #base_url="https://api.openai.com/v1/chat/completions")
    elif "claude" in model_name:
        llm = ChatAnthropic(
            model=model_name,#"claude-3-5-sonnet-20240620", # claude-2.1, claude-3-opus-20240229, claude-3-5-sonnet-20240620
            api_key=api_key,
            temperature=0,
            timeout=300,
        )
    elif "gemini" in model_name:
        llm = ChatGoogleGenerativeAI(
            model=model_name,#"gemini-2.5-pro",
            google_api_key=api_key,
            temperature=0,
            timeout=300,
        )
    else:
        base_url = f"http://{main_ip}:6000/v1"
        print(base_url)
        llm = ChatOpenAI(
            model_name=model_name,
            base_url=base_url,
            api_key="EMPTY")
    LLM = llm
