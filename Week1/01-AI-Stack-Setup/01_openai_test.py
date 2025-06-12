# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# TEST OPENAI AND SET UP CREDENTIALS
# ***

# Goals:
# 1. Setup OpenAI API Credentials
# 2. Test API connection


# 1.0 OPENAI API ACCESS

#  1. Sign Up or Log In: First, you need to create an account with OpenAI or log into an existing account. Visit OpenAI's API page for this.

#  2. API Key: Once logged in, you will need to access the API key management section to obtain your API key. This key is used to authenticate requests to OpenAI's services.

#  3. Set Rate Limits: Soft and hard limits can be set. Soft will send an email when usage limit has been exceeded. Hard will stop the API from running. 

#  4. Secure Your API Key: We will use a simple YAML file (credentials.yml example is provided).   

# 2.0 TEST API CONNECTION

from openai import OpenAI
from dotenv import load_dotenv
import os

print('Hello world')

# Load from the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-3.5-turbo", 
    messages=[
        {"role": "system", "content": "You are john perry from old man's war"},
        {"role": "user", "content": "Tell me something he'd say at the CDF mealtime with other CDF friends"},
    ],
    max_tokens=60
)

print(response.choices[0].message.content)      