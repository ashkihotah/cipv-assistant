from datetime import datetime
import time
import random
from google.genai import types
from google import genai
from tqdm import tqdm
import os
import re

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

N_PERSONAS = 10
CHOOSEN_MODEL = "gemini-2.5-flash"
DATASET_PATH = "./rsc/" + CHOOSEN_MODEL + "-dataset" + "_" + timestamp
# DATASET_PATH = "./rsc/gemini-2.5-flash-dataset_2025-07-07-10-45-16"

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
    },
    "gemini-2.0-flash": {
        "RPM": 15,       # Requests per minute
        "TPM": 1000000,  # Tokens per minute
        "RPD": 200       # Requests per day
    },
    "gemini-2.0-flash-lite": {
        "RPM": 30,      # Requests per minute
        "TPM": 1000000,  # Tokens per minute
        "RPD": 200      # Requests per day
    }
}

class DatasetGenerator:
    def __init__(self, dataset_path=DATASET_PATH, choosen_model=CHOOSEN_MODEL):
        self.dataset_path = dataset_path
        self.choosen_model = choosen_model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not set.")
            print("Please set it before running the script!")
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        self.client = genai.Client()
        self.rpm = 0
        os.makedirs(self.dataset_path, exist_ok=True)

    
    def generate_personas(self):
        with open("./rsc/prompts/generate_personas.md", "r") as f:
            os.makedirs(self.dataset_path + "/personas", exist_ok=True)
            personas_guidelines = f.read()
            chat = self.client.chats.create(
                model=self.choosen_model,
                config=types.GenerateContentConfig(
                    system_instruction=personas_guidelines
                )
            )
            for _ in tqdm(range(N_PERSONAS), desc="Generating Personas"):
                response = chat.send_message(
                    "generate"
                )
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                path = f"{self.dataset_path}/personas/{timestamp}.md"
                with open(path, "w", encoding="utf-8") as response_file:
                    response_file.write(response.text)
                self.rpm += 1

    def generate_chat(self, chat, polarity_mean, polarity_variance, persona, stage, polarity):
        persona = persona.split(".")[0]  # Remove the file extension
        os.makedirs(f"{self.dataset_path}/chats/{persona}", exist_ok=True)
        path = f"{self.dataset_path}/chats/{persona}/{stage}_{polarity}.txt"
        if not os.path.exists(path):
            self.rpm += 1
            if self.rpm > rate_limits[self.choosen_model]["RPM"]:
                print(f"Rate limit reached for {self.choosen_model}. Waiting for 60 seconds...")
                time.sleep(60)
                self.rpm = 0
            response = chat.send_message(f"Stage: {stage}\nChat Polarity Mean: {polarity_mean}\nChat Polarity Variance: {polarity_variance}")
            response = f"Stage: {stage}\nChat Polarity Mean: {polarity_mean}\nChat Polarity Variance: {polarity_variance}\n" + response.text
            with open(path, "w", encoding="utf-8") as response_file:
                response_file.write(response)
        else:
            print(f"Chat file {path} already exists. Skipping generation!")

    def generate_chats(self):
        os.makedirs(os.path.dirname(f"{self.dataset_path}/chats/"), exist_ok=True)
        stages = extract_key_stages()
        with open("./rsc/prompts/generate_chat.md", "r") as f1:
            system_prompt = f1.read()
            personas = os.listdir(self.dataset_path + "/personas")
            for persona in personas:
                with open(self.dataset_path + "/personas/" + persona, "r") as f2:
                    persona_content = f2.read()
                    chat = self.client.chats.create(
                        model=self.choosen_model,
                        config=types.GenerateContentConfig(
                            system_instruction= f"{system_prompt}\n{persona_content}"
                        )
                    )
                    for stage in tqdm(stages, desc=f"Generating Chats for couple of personas in {persona}"):
                        self.generate_chat(
                            chat,
                            round(random.uniform(0.5, 1), 2),
                            round(random.uniform(0, 1), 2),
                            persona,
                            stage,
                            "healthy"
                        )
                        self.generate_chat(
                            chat,
                            round(random.uniform(0, 0.5), 2),
                            round(random.uniform(0, 1), 2),
                            persona,
                            stage,
                            "neutral_healthy"
                        )
                        self.generate_chat(
                            chat,
                            round(random.uniform(-0.5, 0), 2),
                            round(random.uniform(0, 1), 2),
                            persona,
                            stage,
                            "neutral_toxic"
                        )
                        self.generate_chat(
                            chat,
                            round(random.uniform(-1, -0.5), 2),
                            round(random.uniform(0, 1), 2),
                            persona,
                            stage,
                            "toxic"
                        )

def extract_key_stages():
    with open("./rsc/prompts/partial/relationship_guidelines.md", "r") as f:
        content = f.read()
        stages = re.findall(r"\#\# .* Stage: (?P<stage>.+)\n", content)
        return stages

generator = DatasetGenerator()
generator.generate_personas(DATASET_PATH)
generator.generate_chats()
