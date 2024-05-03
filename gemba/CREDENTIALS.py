# deployment is used to list available models
# for Azure API, specify model name as a key and deployment name as a value
# for OpenAI API, specify model name as a key and a value
import os
credentials = {
    "deployments": {"gpt-4-turbo-2024-04-09": "gpt-4-turbo-2024-04-09"},
    "api_key": os.getenv("OPENAI_API_KEY"),
    "requests_per_second_limit": 1
}