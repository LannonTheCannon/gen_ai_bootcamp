# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# BONUS: HOW TO RUN LOCALL LLMS FOR DATA PRIVACY AND FREE LLM INFERENCE
# ***
# GOAL: Get Ollama set up to run local LLMs

# NOTE - This tutorial should be run AFTER completing the AI Fast Track because it depends on having LangChain installed


# STEP 1: ACTIVATE ds4b_301p_dev ENVIRONMENT

# STEP 2: DOWNLOAD OLLAMA
#  https://www.ollama.com/download

# STEP 2A: PICK AN OPEN MODEL TO DOWNLOAD
#  https://www.ollama.com/library 

# STEP 3: INSTALL llama3.2:3b Model 
# NOTE: TAKES SEVERAL MINUTES TO DOWNLOAD
# * ollama run llama3.2:3b

#   * Models are stored in ~/.ollama folder
#   * Ctrl + D to exit the interactive client

# STEP 4: INSTALL PYTHON OLLAMA PACKAGE
#  pip install ollama langchain-ollama

# STEP 5: USE LANGCHAIN

from langchain_ollama.chat_models import ChatOllama

import re
from IPython.display import Markdown
from pprint import pprint

# LLAMA 3.2:3B
# * ollama run llama3.2:3b
# * Ctrl + D to exit the interactive client

llm = ChatOllama(model="llama3.2:3b")
llm

response = llm.invoke("what color is the sky normally? Use one word response.")

response.content

response = llm.invoke("What's the recipe for mayonaise?")

Markdown(response.content)


# DEEPSEEK AI: Took 5 minutes to download
# * ollama run deepseek-r1
# * Ctrl + D to exit the interactive client

llm = ChatOllama(model="deepseek-r1:latest")
llm

response = llm.invoke("What's the best place to get pizza in Pittsburgh, PA?")

pprint(response.content)

Markdown(response.content)

# * DEEPSEEK AI: This is a Chain-Of-Thought (CoT) model (often called a "reasoning" model)

# * Returns <think>...</think> block

# * NEW: STRIP THINKING:

def strip_thinking(content: str) -> str:
    # Remove the <think>...</think> block
    return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)

pprint(strip_thinking(response.content))

Markdown(strip_thinking(response.content))

# CONCLUDING COMMENTS:
# 1. WE ARE USING THE LOCAL LLM
# 2. INFERENCE SPEED CAN BE SLOWER FOR LONGER RESPONSES. 
# 3. SPEEDING UP INFERENCING TYPICALLY REQUIRES GPUS
# 4. NOT ALL USERS HAVE ACCESS TO LOCAL LLMS OR GPUS
# 5. CLOUD PLATFORMS LIKE AWS OFFER GPU-BASED SERVICES LIKE AMAZON BEDROCK (https://aws.amazon.com/bedrock/)
