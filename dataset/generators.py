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

class Generator:

    def __init__(self, choosen_model):
        self.choosen_model = choosen_model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not set.")
            print("Please set it before running the script!")
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        self.client = genai.Client()
        self.rpm = 0
        self.tpm = 0
    
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

class CouplesGenerator(Generator):
    def __init__(self, choosen_model, lang="ITA"):
        super().__init__(choosen_model=choosen_model)
        self.prompts_path = os.path.join(".", "rsc", "prompts", lang)
    
    def get_couples_prompt(self):
        path = os.path.join(self.prompts_path, f"generate_couples.md")
        with open(path, "r", encoding="utf-8") as f1:
            path = os.path.join(self.prompts_path, "partial", "personas_guidelines.md")
            with open(path, "r", encoding="utf-8") as f2:
                return f1.read() + '\n' + f2.read()
        raise Exception("Error reading couples prompt file!")

    def generate(self, n_couples, out_path):
        couples_guidelines = self.get_couples_prompt()
        os.makedirs(out_path, exist_ok=True)
        self.update_rate_limit(couples_guidelines)
        chat = self.client.chats.create(
            model=self.choosen_model,
            config=types.GenerateContentConfig(
                system_instruction=couples_guidelines
            )
        )
        for _ in tqdm(range(n_couples), desc="Generating Couples"):
            prompt = "generate"
            self.update_rate_limit(prompt)
            response = chat.send_message(
                prompt
            )
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            path = os.path.join(out_path, f"{timestamp}.md")
            with open(path, "w", encoding="utf-8") as response_file:
                response_file.write(response.text)

class CouplesToChatGenerator(Generator):
    def __init__(self, choosen_model, couples_path, lang="ITA"):
        super().__init__(choosen_model=choosen_model)
        self.prompts_path = os.path.join(".", "rsc", "prompts", lang)
        self.couples_path = couples_path

    def get_chats_prompt(self):
        path = os.path.join(self.prompts_path, f"generate_chat.md")
        with open(path, "r", encoding="utf-8") as f1:
            path = os.path.join(self.prompts_path, "partial", f"relationship_guidelines.md")
            with open(path, "r", encoding="utf-8") as f2:
                return f1.read() + '\n' + f2.read()

    def generate_chat(self, out_path, chat, polarity_mean, polarity_variance, couple, stage, polarity):
        couple = couple.split(".")[0]  # Remove the file extension
        dir_path = os.path.join(out_path, couple)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{stage} ({polarity}).txt")
        if not os.path.exists(path):
            prompt = f"Stage: {stage}\nChat Polarity Mean: {polarity_mean}\nChat Polarity Variance: {polarity_variance}"
            self.update_rate_limit(prompt)
            response = chat.send_message(prompt)
            response = f"Stage: {stage}\nChat Polarity Mean: {polarity_mean}\nChat Polarity Variance: {polarity_variance}\n" + response.text
            with open(path, "w", encoding="utf-8") as response_file:
                response_file.write(response)
        # else:
        #     print(f"Chat file {path} already exists. Skipping generation!")

    def generate(self, out_path):
        chat_prompt = self.get_chats_prompt()
        os.makedirs(out_path, exist_ok=True)
        stages = self.extract_key_stages()
        couples = os.listdir(self.couples_path)
        for couple in couples:
            with open(os.path.join(self.couples_path, couple), "r", encoding="utf-8") as f2:
                couple_content = f2.read()
                prompt = f"{chat_prompt}\n{couple_content}"
                # self.update_rate_limit(prompt)
                chat = self.client.chats.create(
                    model=self.choosen_model,
                    config=types.GenerateContentConfig(
                        system_instruction=prompt
                    )
                )
                for stage in tqdm(stages, desc=f"Generating Chats for couple {couple}"):
                    self.generate_chat(
                        out_path,
                        chat,
                        round(random.uniform(0.35, 1), 2),
                        round(random.uniform(0, 1), 2),
                        couple,
                        stage,
                        "healthy"
                    )
                    self.generate_chat(
                        out_path,
                        chat,
                        round(random.uniform(-0.35, 0.35), 2),
                        round(random.uniform(0, 1), 2),
                        couple,
                        stage,
                        "neutral"
                    )
                    self.generate_chat(
                            out_path,
                            chat,
                            round(random.uniform(-1, -0.35), 2),
                            round(random.uniform(0, 1), 2),
                            couple,
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

    N_PERSONAS = 10
    CHOOSEN_MODEL = "gemini-2.5-pro"

    OUT_DIR = os.path.join(".", "out", "datasets", "gen2")
    # SUFFIX_PATH = CHOOSEN_MODEL + "_" + timestamp
    SUFFIX_PATH = "gemini-2.5-pro_2025-07-25-13-12-37"

    COUPLES_PATH = os.path.join(OUT_DIR, "couples", SUFFIX_PATH)
    CHATS_PATH = os.path.join(OUT_DIR, "chats", SUFFIX_PATH)

    couples_generator = CouplesGenerator(
        choosen_model=CHOOSEN_MODEL,
        lang="ITA"  # Change to "ENG" for English
    )

    chats_generator = CouplesToChatGenerator(
        couples_path=COUPLES_PATH,
        choosen_model=CHOOSEN_MODEL,
        lang="ITA"  # Change to "ENG" for English
    )

    # couples_generator.generate(N_PERSONAS, COUPLES_PATH)
    chats_generator.generate(CHATS_PATH)
