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

class RpdLimitExceeded(Exception):
    """Exception raised when the RPD limit is exceeded."""
    def __init__(self, message="RPD limit exceeded. Please try again tomorrow."):
        self.message = message
        super().__init__(self.message)

class LimitsHandler:

    def __init__(self, client, choosen_model):
        self.choosen_model = choosen_model
        self.client = client
        self.rpm = 0
        self.tpm = 0
        self.rpd = 0
        self.last_reset_time = datetime.now()
    
    def update_rate_limit(self, prompt):
        token_count = self.client.models.count_tokens(
            model=self.choosen_model, contents=prompt
        )
        self.tpm += token_count.total_tokens
        self.rpm += 1
        is_minute_passed = (datetime.now() - self.last_reset_time).total_seconds() >= 60
        if is_minute_passed:
            if self.rpm > rate_limits[self.choosen_model]["RPM"]:
                print(f"RPM limit reached for {self.choosen_model}. Waiting for 60 seconds...")
                time.sleep(60)
                self.rpm = 1
            elif self.tpm > rate_limits[self.choosen_model]["TPM"]:
                print(f"TPM limit reached for {self.choosen_model}. Waiting for 60 seconds...")
                time.sleep(60)
                self.tpm = token_count
            elif self.rpd > rate_limits[self.choosen_model]["RPD"]:
                print(f"RPD limit reached for {self.choosen_model}.")
                raise RpdLimitExceeded()
            self.last_reset_time = datetime.now()

class CouplesGenerator():
    def __init__(
            self, client, choosen_model, lang="ITA",
            limits_handler=None
        ):
        self.choosen_model = choosen_model
        self.client = client
        if limits_handler is None:
            self.limits_handler = LimitsHandler(client, choosen_model)
        else:
            self.limits_handler = limits_handler
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
        self.limits_handler.update_rate_limit(couples_guidelines)
        chat = self.client.chats.create(
            model=self.choosen_model,
            config=types.GenerateContentConfig(
                system_instruction=couples_guidelines
            )
        )
        for _ in tqdm(range(n_couples), desc="Generating Couples"):
            prompt = "generate"
            self.limits_handler.update_rate_limit(prompt)
            response = chat.send_message(
                prompt
            )
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            path = os.path.join(out_path, f"{timestamp}.md")
            with open(path, "w", encoding="utf-8") as response_file:
                response_file.write(response.text)

class CouplesToChatGenerator():
    def __init__(
            self, client, choosen_model,
            couples_path, lang="ITA",
            limits_handler=None,
            pred_type="sentiment"

        ):
        self.choosen_model = choosen_model
        self.client = client
        if limits_handler is None:
            self.limits_handler = LimitsHandler(client, choosen_model)
        else:
            self.limits_handler = limits_handler
        self.prompts_path = os.path.join(".", "rsc", "prompts", lang)
        self.couples_path = couples_path
        self.chat_prompt = self.get_chats_prompt(pred_type=pred_type)

    def get_chats_prompt(self, pred_type="sentiment"):
        path = os.path.join(self.prompts_path, f"generate_chat_{pred_type}.md")
        with open(path, "r", encoding="utf-8") as f1:
            path = os.path.join(self.prompts_path, "partial", f"relationship_guidelines.md")
            with open(path, "r", encoding="utf-8") as f2:
                return f1.read() + '\n' + f2.read()

    def generate_chat(self, out_path, chat, polarity_mean, couple, stage, polarity):
        couple = couple.split(".")[0]  # Remove the file extension
        dir_path = os.path.join(out_path, couple)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{stage} ({polarity}).txt")
        if not os.path.exists(path):
            prompt = f"Stage: {stage}\nChat Polarity: {polarity_mean}\n"
            self.limits_handler.update_rate_limit(prompt)
            response = chat.send_message(prompt)
            response = f"Stage: {stage}\nChat Polarity: {polarity_mean}\n" + response.text
            with open(path, "w", encoding="utf-8") as response_file:
                response_file.write(response)
        # else:
        #     print(f"Chat file {path} already exists. Skipping generation!")

    def generate(self, out_path):
        os.makedirs(out_path, exist_ok=True)
        stages = self.extract_key_stages()
        couples = os.listdir(self.couples_path)
        for couple in couples:
            with open(os.path.join(self.couples_path, couple), "r", encoding="utf-8") as f2:
                couple_content = f2.read()
                prompt = f"{self.chat_prompt}\n{couple_content}"
                # self.limits_handler.update_rate_limit(prompt)
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
                        # round(random.uniform(0.35, 1), 2),
                        "healthy",
                        couple,
                        stage,
                        "healthy"
                    )
                    self.generate_chat(
                        out_path,
                        chat,
                        # round(random.uniform(-0.35, 0.35), 2),
                        "neutral",
                        couple,
                        stage,
                        "neutral"
                    )
                    self.generate_chat(
                            out_path,
                            chat,
                            # round(random.uniform(-1, -0.35), 2),
                            "toxic",
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

class ChatToExplanationGenerator():
    def __init__(
            self, client, choosen_model, chat_path,
            lang="ITA", limits_handler=None,
            pred_type="sentiment"
        ):
        self.choosen_model = choosen_model
        self.client = client
        if limits_handler is None:
            self.limits_handler = LimitsHandler(client, choosen_model)
        else:
            self.limits_handler = limits_handler
        
        self.prompts_path = os.path.join(".", "rsc", "prompts", lang)
        self.chat_path = chat_path
        self.explanation_prompt = self.get_explanation_prompt(pred_type=pred_type)
        self.skipped = 0

    def get_explanation_prompt(self, pred_type="sentiment"):
        path = os.path.join(self.prompts_path, f"generate_explanation_{pred_type}.md")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_chat(self, chat_content):
        msgs_regex = re.compile(r"(?P<message>(?P<msg_no_polarity>\(?(?P<timestamp>\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d)\)? ?\|? ?(?P<name_content>(?P<name>.+):\n?\s?(?P<content>.+)))\n?\s?Polarity: (?P<polarity>(?:-?|\+?)\d\.?\d?\d?))")
        messages = list(msgs_regex.finditer(chat_content))
        if len(messages) == 0:
            self.skipped += 1
            return None
        all_messages = [msg.group("msg_no_polarity") + f"\nPolarity: {msg.group('polarity')}" for msg in messages]
        return "\n\n".join(all_messages)

    def generate(self, out_path):
        os.makedirs(out_path, exist_ok=True)
        couples = os.listdir(self.chat_path)
        for couple in couples:
            in_couple_dir = os.path.join(self.chat_path, couple)
            out_couple_dir = os.path.join(out_path, couple)
            os.makedirs(out_couple_dir, exist_ok=True)
            chats = os.listdir(in_couple_dir)
            for chat_file in tqdm(chats, desc="Generating Explanations"):
                with open(os.path.join(in_couple_dir, chat_file), "r", encoding="utf-8") as f:
                    
                    chat_content = self.extract_chat(f.read())
                    path = os.path.join(out_couple_dir, f"{chat_file}")
                    if os.path.exists(path) or chat_content is None:
                        # print(f"Explanation file {path} already exists. Skipping generation!")
                        continue

                    prompt = f"{self.explanation_prompt}\n{chat_content}"
                    self.limits_handler.update_rate_limit(self.explanation_prompt + prompt)
                    response = self.client.models.generate_content(
                        model=self.choosen_model,
                        config=types.GenerateContentConfig(
                            system_instruction=self.explanation_prompt
                        ),
                        contents=chat_content
                    )
                    with open(path, "w", encoding="utf-8") as response_file:
                        response_file.write(f"{chat_content}\n\n{response.text}")
        print(f"Skipped {self.skipped} chats due to no messages found.")

class CouplesToExplainedChatGenerator():
    def __init__(
            self, client, choosen_model, couples_path,
            lang="ITA", limits_handler=None,
            pred_type="sentiment"
        ):
        self.choosen_model = choosen_model
        self.client = client
        if limits_handler is None:
            self.limits_handler = LimitsHandler(client, choosen_model)
        else:
            self.limits_handler = limits_handler
        
        self.prompts_path = os.path.join(".", "rsc", "prompts", lang)
        self.couples_path = couples_path
        self.explanation_prompt = self.get_explanation_prompt(pred_type=pred_type)
        self.chat_prompt = self.get_chats_prompt(pred_type=pred_type)

    def get_explanation_prompt(self, pred_type="sentiment"):
        path = os.path.join(self.prompts_path, f"generate_explanation_{pred_type}.md")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def get_chats_prompt(self, pred_type="sentiment"):
        path = os.path.join(self.prompts_path, f"generate_chat_{pred_type}.md")
        with open(path, "r", encoding="utf-8") as f1:
            path = os.path.join(self.prompts_path, "partial", f"relationship_guidelines.md")
            with open(path, "r", encoding="utf-8") as f2:
                return f1.read() + '\n' + f2.read()

    def generate_chat(self, out_path, chat, polarity_mean, couple, stage, polarity):
        couple = couple.split(".")[0]  # Remove the file extension
        dir_path = os.path.join(out_path, couple)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{stage} ({polarity}).txt")
        if not os.path.exists(path):
            prompt = f"Stage: {stage}\nChat Polarity: {polarity_mean}\n"
            self.limits_handler.update_rate_limit(prompt)
            response = chat.send_message(prompt)

            chat = response.text # self.extract_chat(response.text)

            self.limits_handler.update_rate_limit(self.explanation_prompt + chat)
            explanation = self.client.models.generate_content(
                model=self.choosen_model,
                config=types.GenerateContentConfig(
                    system_instruction=self.explanation_prompt
                ),
                contents=chat
            )

            response = response.text # f"Stage: {stage}\nChat Polarity: {polarity_mean}\n"
            response += f"\n\n{explanation.text}"
            with open(path, "w", encoding="utf-8") as response_file:
                response_file.write(response)

    # def extract_chat(self, chat_content):
    #     msgs_regex = re.compile(r"(?P<message>(?P<msg_no_polarity>\(?(?P<timestamp>\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d)\)? ?\|? ?(?P<name_content>(?P<name>.+):\n?\s?(?P<content>.+)))\n?\s?Polarity: (?P<polarity>(?:-?|\+?)\d\.?\d?\d?))")
    #     messages = list(msgs_regex.finditer(chat_content))
    #     if len(messages) == 0:
    #         return chat_content
    #     all_messages = [msg.group("msg_no_polarity") for msg in messages]
    #     return "\n\n".join(all_messages)

    def generate(self, out_path):
        os.makedirs(out_path, exist_ok=True)
        stages = self.extract_key_stages()
        couples = os.listdir(self.couples_path)
        for couple in couples:
            with open(os.path.join(self.couples_path, couple), "r", encoding="utf-8") as f2:
                couple_content = f2.read()
                prompt = f"{self.chat_prompt}\n{couple_content}"
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
                        # round(random.uniform(0.35, 1), 2),
                        "healthy",
                        couple,
                        stage,
                        "healthy"
                    )
                    self.generate_chat(
                        out_path,
                        chat,
                        # round(random.uniform(-0.35, 0.35), 2),
                        "neutral",
                        couple,
                        stage,
                        "neutral"
                    )
                    self.generate_chat(
                            out_path,
                            chat,
                            # round(random.uniform(-1, -0.35), 2),
                            "toxic",
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

    N_PERSONAS = 2
    CHOOSEN_MODEL = "gemini-2.5-flash"
    API_KEY_ENV_VAR = "GEMINI_API_KEY_1"
    PRED_TYPE = "toxicity"  # Change to "sentiment" or "toxicity" as needed
    SUFFIX_PATH = timestamp + '_' + CHOOSEN_MODEL
    # SUFFIX_PATH = "gemini-2.0-flash-dataset_2025-07-09-15-56-22"
    OUT_DIR = os.path.join(".", "out", "datasets", PRED_TYPE, "gen1")
    os.makedirs(OUT_DIR, exist_ok=True)

    COUPLES_PATH = os.path.join(OUT_DIR, "couples", SUFFIX_PATH)
    CHATS_PATH = os.path.join(OUT_DIR, "chats", SUFFIX_PATH)
    EXPLANATIONS_PATH = os.path.join(OUT_DIR, "explained", SUFFIX_PATH)

    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise ValueError(f"Environment variable '{API_KEY_ENV_VAR}' not set.")
    print(f"Using API key env var: {API_KEY_ENV_VAR}")
    print(f"with value: {api_key}")
    client = genai.Client(api_key=api_key)

    limits_handler = LimitsHandler(client, CHOOSEN_MODEL)

    couples_generator = CouplesGenerator(
        choosen_model=CHOOSEN_MODEL,
        lang="ITA",
        client=client,
        limits_handler=limits_handler
    )
    couples_generator.generate(N_PERSONAS, COUPLES_PATH)

    chats_generator = CouplesToExplainedChatGenerator(
        couples_path=COUPLES_PATH,
        choosen_model=CHOOSEN_MODEL,
        lang="ITA",
        client=client,
        pred_type=PRED_TYPE,
        limits_handler=limits_handler
    )
    chats_generator.generate(CHATS_PATH)

    # explanations_generator = ChatToExplanationGenerator(
    #     chat_path=CHATS_PATH,
    #     choosen_model=CHOOSEN_MODEL,
    #     lang="ITA",
    #     client=client,
    #     pred_type=PRED_TYPE,
    #     limits_handler=limits_handler
    # )
    # explanations_generator.generate(EXPLANATIONS_PATH)
