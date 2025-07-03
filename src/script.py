import os
from google import genai
from datetime import datetime
# import os

N_PERSONAS = 5
N_CHATS = 3

PERSONAS_OUT_PATH = "./rsc/personas/"
CHATS_OUT_PATH = "./rsc/chats/"
DATASET_PATH = "./rsc/dataset"

CHOOSEN_MODEL = "gemini-2.5-flash-lite-preview-06-17"

rate_limits = {
    "gemini-2.5-pro": {
        "RPM": 5,       # Requests per minute
        "TPM": 250000,  # Tokens per minute
        "RPD": 100      # Requests per day
    },
    "gemini-2.5-flash": {
        "RPM": 10,      # Requests per minute
        "TPM": 250000,  # Tokens per minute
        "RPD": 250      # Requests per day
    },
    "gemini-2.5-flash-lite-preview-06-17": {
        "RPM": 15,      # Requests per minute
        "TPM": 250000,  # Tokens per minute
        "RPD": 1000     # Requests per day
    }
}

json_schema = {
    "description": str,
    "embedding": list[float],
    "traits": {
        "openness": int,
        "conscientiousness": int,
        "extraversion": int,
        "agreeableness": int,
        "neuroticism": int
    },
    "age": int,
    "location": str,
    "gender": str,
    "education_level": int

}

# Uncomment the following lines if you want to use an API key from an environment variable
# api_key = os.getenv("GEMINI_API_KEY")
# client = genai.Client(api_key=api_key)

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

def generate_personas(dataset_path):
    chat = client.chats.create(model=CHOOSEN_MODEL)
    with open("./rsc/ENG_personas_prompt.md", "r") as f:
        personas_prompt = f.read()
        for _ in range(N_PERSONAS):
            response = chat.send_message(
                personas_prompt
            )
            # Write the response to a file named as persona_{timestamp}.txt
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            with open(f"{dataset_path}{PERSONAS_OUT_PATH}{timestamp}.txt", "w", encoding="utf-8") as f:
                f.write(response.text)

def generate_chats():
    chat = client.chats.create(model=CHOOSEN_MODEL)
    with open("./rsc/ENG_chat_prompt.md", "r") as f:
        chat_prompt = f.read()
        for _ in range(N_CHATS):
            response = chat.send_message(
                chat_prompt
            )
            # Write the response to a file named as chat_{timestamp}.txt
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            with open(f"{CHATS_OUT_PATH}{timestamp}.txt", "w", encoding="utf-8") as f:
                f.write(response.text)

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
dataset_path = DATASET_PATH + "_" + timestamp + "/"
os.makedirs(dataset_path)
generate_personas(dataset_path)
