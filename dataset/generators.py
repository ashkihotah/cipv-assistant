from datetime import datetime
import time
import random
from google.genai import types
from google import genai
from tqdm.auto import tqdm
import os
import re

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
    def __init__(self, prefix_path, suffix_path, choosen_model, lang="ITA"):
        self.prefix_path = prefix_path
        self.suffix_path = suffix_path
        self.choosen_model = choosen_model
        self.prompts_path = os.path.join(".", "rsc", "prompts", "n_polarities_to_1_explanation", lang)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not set.")
            print("Please set it before running the script!")
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        self.client = genai.Client()
        self.rpm = 0
        self.tpm = 0
        os.makedirs(self.prefix_path, exist_ok=True)
    
    def get_personas_prompt(self):
        path = os.path.join(self.prompts_path, f"generate_personas.md")
        with open(path, "r", encoding="utf-8") as f1:
            path = os.path.join(self.prompts_path, "partial", f"personas_guidelines.md")
            with open(path, "r", encoding="utf-8") as f2:
                personas_guidelines = f1.read()
                personas_guidelines += f2.read()
                return personas_guidelines
        raise Exception("Error reading personas prompt file!")

    def get_chats_prompt(self):
        path = os.path.join(self.prompts_path, f"generate_chat.md")
        with open(path, "r", encoding="utf-8") as f1:
            path = os.path.join(self.prompts_path, "partial", f"relationship_guidelines.md")
            with open(path, "r", encoding="utf-8") as f2:
                chats_prompt = f1.read()
                chats_prompt += f2.read()
                return chats_prompt

    def generate_personas(self, n_personas):
        personas_guidelines = self.get_personas_prompt()
        partial_path = os.path.join(self.prefix_path, "personas")
        os.makedirs(partial_path, exist_ok=True)
        total_path = os.path.join(partial_path, self.suffix_path)
        os.makedirs(total_path, exist_ok=True)
        self.update_rate_limit(personas_guidelines)
        chat = self.client.chats.create(
            model=self.choosen_model,
            config=types.GenerateContentConfig(
                system_instruction=personas_guidelines
            )
        )
        for _ in tqdm(range(n_personas), desc="Generating Personas"):
            prompt = "generate"
            self.update_rate_limit(prompt)
            response = chat.send_message(
                prompt
            )
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            path = os.path.join(total_path, f"{timestamp}.md")
            with open(path, "w", encoding="utf-8") as response_file:
                response_file.write(response.text)

    def update_rate_limit(self, prompt):
        token_count = self.client.models.count_tokens(
            model=self.choosen_model, contents=prompt
        )
        self.tpm += token_count.total_tokens
        if self.rpm > rate_limits[self.choosen_model]["RPM"]:
            print(f"RPM limit reached for {self.choosen_model}. Waiting for 60 seconds...")
            time.sleep(60)
            self.rpm = 1
        elif self.tpm > rate_limits[self.choosen_model]["TPM"]:
            print(f"TPM limit reached for {self.choosen_model}. Waiting for 60 seconds...")
            time.sleep(60)
            self.tpm = token_count
        else:
            self.rpm += 1

    def generate_chat(self, total_path, chat, polarity_mean, polarity_variance, persona, stage, polarity):
        persona = persona.split(".")[0]  # Remove the file extension
        dir_path = os.path.join(total_path, persona)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{stage} ({polarity}).txt")
        if not os.path.exists(path):
            prompt = f"Stage: {stage}\nChat Polarity Mean: {polarity_mean}\nChat Polarity Variance: {polarity_variance}"
            self.update_rate_limit(prompt)
            response = chat.send_message(prompt)
            response = f"Stage: {stage}\nChat Polarity Mean: {polarity_mean}\nChat Polarity Variance: {polarity_variance}\n" + response.text
            with open(path, "w", encoding="utf-8") as response_file:
                response_file.write(response)
        else:
            print(f"Chat file {path} already exists. Skipping generation!")

    def generate_chats(self):
        system_prompt = self.get_chats_prompt()
        partial_path = os.path.join(self.prefix_path, "chats")
        os.makedirs(partial_path, exist_ok=True)
        total_path = os.path.join(partial_path, self.suffix_path)
        os.makedirs(total_path, exist_ok=True)
        stages = self.extract_key_stages()
        personas_path = os.path.join(self.prefix_path, "personas", self.suffix_path)
        personas = os.listdir(personas_path)
        for persona in personas:
            with open(os.path.join(personas_path, persona), "r", encoding="utf-8") as f2:
                persona_content = f2.read()
                prompt = f"{system_prompt}\n{persona_content}"
                # self.update_rate_limit(prompt)
                chat = self.client.chats.create(
                    model=self.choosen_model,
                    config=types.GenerateContentConfig(
                        system_instruction=prompt
                    )
                )
                for stage in tqdm(stages, desc=f"Generating Chats for couple of personas in {persona}"):
                    self.generate_chat(
                        total_path,
                        chat,
                        round(random.uniform(0.35, 1), 2),
                        round(random.uniform(0, 1), 2),
                        persona,
                        stage,
                        "healthy"
                    )
                    self.generate_chat(
                        total_path,
                        chat,
                        round(random.uniform(-0.35, 0.35), 2),
                        round(random.uniform(0, 1), 2),
                        persona,
                        stage,
                        "neutral"
                    )
                    self.generate_chat(
                            total_path,
                            chat,
                            round(random.uniform(-1, -0.35), 2),
                            round(random.uniform(0, 1), 2),
                            persona,
                            stage,
                            "toxic"
                        )

    def extract_key_stages(self):
        path = os.path.join(self.prompts_path, "partial", f"relationship_guidelines.md")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            stages = re.findall(r"\#\# (?P<stage>[\w -]+)\n", content)
            return stages

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    N_PERSONAS = 17
    CHOOSEN_MODEL = "gemini-2.5-flash"

    PREFIX_PATH = os.path.join(".", "out", "datasets", "gen2")
    # SUFFIX_PATH = CHOOSEN_MODEL + "_" + timestamp
    SUFFIX_PATH = "gemini-2.5-flash_2025-07-23-17-15-50"  # Use a fixed suffix for testing

    generator = DatasetGenerator(
        prefix_path=PREFIX_PATH,
        suffix_path=SUFFIX_PATH,
        choosen_model=CHOOSEN_MODEL,
        lang="ITA"  # Change to "ENG" for English
    ) 
    print(generator.extract_key_stages())
    # generator.generate_personas(N_PERSONAS)
    # generator.generate_chats()
