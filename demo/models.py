import joblib
from nltk.stem.snowball import SnowballStemmer
import spacy
import os

import torch

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BertForSequenceClassification,
    GenerationConfig
)

from nlp_for_ml.preprocessing import (
    SpacyPosNerPreprocessor,
    SpacyPosNerPreprocessorWithStemming
)

# from transformers_for_chat_analysis.bert import BertForMultipleRegressions
# from transformers_for_chat_analysis.multi_task_bart import (
#     BartForConditionalGenerationAndRegression
# )

BINARY_LABELS = ['Toxic', 'Healthy']
MULTICLASS_LABELS = ['Toxic', 'Neutral', 'Healthy']

def decode_label(label_id: int, labels: str) -> str:
    if labels == 'binary':
        return BINARY_LABELS[label_id]
    elif labels == 'multiclass':
        return MULTICLASS_LABELS[label_id]
    raise ValueError("Unknown labels type")

class MLChatClassifier:

    MODEL_DIR_PREFIX = os.path.join(".", "out", "models")

    def __init_preprocessor(self, preprocessor: str):
        if preprocessor == "SpacyPosNerPreprocessorWithStemming":
            self.preprocessor = SpacyPosNerPreprocessorWithStemming(
                nlp=spacy.load("it_core_news_sm"),
                stemmer=SnowballStemmer("italian"),
                ner_to_replace=['PERSON', 'ORG', 'LOC'],
                pos_whitelist=['VERB', 'ADJ', 'ADV', 'NOUN', 'INTJ', 'PRON', 'AUX']
            )
        elif preprocessor == "SpacyPosNerPreprocessor":
            self.preprocessor = SpacyPosNerPreprocessor(
                nlp=spacy.load("it_core_news_sm"),
                ner_to_replace=['PERSON', 'ORG', 'LOC'],
                pos_whitelist=['VERB', 'ADJ', 'ADV', 'NOUN', 'INTJ', 'PRON', 'AUX']
            )

    def __init__(self):
        self.pipeline = None
        self.loaded_pipeline = None

    def load_pipeline(self, task: str, labels: str, preprocessor: str, model: str):
        pipeline_dir = os.path.join(self.MODEL_DIR_PREFIX, task)
        pipeline_dir = os.path.join(pipeline_dir, f"entire-chat-{labels}-classification")
        pdirs = sorted([pdir for pdir in os.listdir(pipeline_dir) if preprocessor in pdir])
        pipeline_dir = os.path.join(pipeline_dir, pdirs[0], model)
        print(f"Loading pipeline from: {pipeline_dir}")
        pipeline_dir = os.path.join(pipeline_dir, "best_model_pipeline.pkl")

        if not os.path.exists(pipeline_dir):
            raise FileNotFoundError(f"Pipeline not found: {pipeline_dir}")

        # load the pipeline with joblib
        if self.loaded_pipeline != pipeline_dir:
            self.loaded_pipeline = pipeline_dir
            self.__init_preprocessor(preprocessor)
            self.pipeline = joblib.load(pipeline_dir)

    def predict(self, chat: str):
        if not self.pipeline:
            raise ValueError("Pipeline not loaded")

        preprocessed = self.preprocessor(chat)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict([preprocessed])[0], self.pipeline.predict_proba([preprocessed])[0]
        else:
            return self.pipeline.predict([preprocessed])[0]

    def load_predict(self, task: str, labels: str, preprocessor: str, model: str, chat: str):
        self.load_pipeline(task, labels, preprocessor, model)
        return self.predict(chat)

class BertChatClassifier:

    def __init__(self, device):
        self.bert = None
        self.tokenizer = None
        self.bert_model = None
        self.device = device

    def load_bert(self, task: str, labels: str, model: str):
        bert_dir = f"./out/models/{task}"
        bert_dir += f"/entire-chat-{labels}-classification/"
        
        mdirs = sorted([mdir for mdir in os.listdir(bert_dir) if model in mdir])
        bert_dir += mdirs[0]

        print(f"Loading BERT model from dir: {bert_dir}")

        last_checkpoint = sorted(
            [f for f in os.listdir(bert_dir) if f.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
            reverse=True
        )[0]
        print(f"Loading BERT model from checkpoint: {last_checkpoint}")

        bert_dir = f"{bert_dir}/{last_checkpoint}"

        if self.bert_model != bert_dir:
            self.bert_model = bert_dir
            self.bert = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
            temp = model.split("-")
            self.with_token_type_ids = temp[-1].split("_")[-1] == 'True'
            print(f"with_token_type_ids: {self.with_token_type_ids}")
            self.with_sep_tokens = temp[-2].split("_")[-1] == 'True'
            print(f"with_sep_tokens: {self.with_sep_tokens}")

    def preprocess(self, msgs):
        out = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        }
        for i, msg in enumerate(msgs):
            if i == 0:
                msg['message'] = f"[CLS]{msg['message']}"
            if self.with_sep_tokens:
                msg['message'] = f"{msg['message']}[SEP]"
            tknzd_msg = self.tokenizer(
                msg['message'] + '\n', # return_tensors="pt",
                max_length=512, truncation=True,
                add_special_tokens=False
            )
            out['input_ids'].extend(tknzd_msg['input_ids'])
            out['attention_mask'].extend(tknzd_msg['attention_mask'])
            if self.with_token_type_ids:
                out['token_type_ids'].extend([msg['person_id']] * len(tknzd_msg['input_ids']))
            else:
                out['token_type_ids'].extend(tknzd_msg['token_type_ids'])
        print(out)
        out = {
            'input_ids': torch.tensor(out['input_ids'], device=self.device).unsqueeze(0),
            'attention_mask': torch.tensor(out['attention_mask'], device=self.device).unsqueeze(0),
            'token_type_ids': torch.tensor(out['token_type_ids'], device=self.device).unsqueeze(0)
        }
        return out

    def predict(self, msgs):
        if not self.bert or not self.tokenizer:
            raise ValueError("BERT model or tokenizer not loaded")

        inputs = self.preprocess(msgs)
        outputs = self.bert(**inputs)
        # return the labels and a list of probabilities
        label_id = outputs.logits.argmax(dim=-1).item()
        probabilities = outputs.logits.softmax(dim=-1).tolist()
        return label_id, probabilities[0]

    def load_predict(self, task: str, labels: str, model: str, msgs: str):
        self.load_bert(task, labels, model)
        return self.predict(msgs)

class BartChatExplanator:

    def __init__(self, device):
        self.bart = None
        self.tokenizer = None
        self.bart_model = None
        self.device = device

    def load_bart(self, task: str, model: str):
        bart_dir = f"./out/models/{task}"
        bart_dir += f"/messages-regression-explanation"
        bart_dir += f"/BART/latest"

        last_checkpoint = sorted(
            [f for f in os.listdir(bart_dir) if f.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
            reverse=True
        )[0]

        bart_dir += f"/{last_checkpoint}"

        if self.bart_model != bart_dir:
            self.bart_model = bart_dir
            self.bart = AutoModelForSeq2SeqLM.from_pretrained(bart_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(bart_dir)
            self.gen_config = GenerationConfig(
                max_length=1024,
                do_sample=True,
                top_p=0.95,
                top_k=25,
                temperature=0.6,
                decoder_start_token_id=self.bart.config.decoder_start_token_id,
                bos_token_id=self.bart.config.bos_token_id,
            )

    def predict(self, text: str):
        if not self.bart or not self.tokenizer:
            raise ValueError("BART model or tokenizer not loaded")

        inputs = self.tokenizer(
            text, return_tensors="pt",
            max_length=1024, truncation=True
        )
        
        # Generate text using the generation config
        generated_ids = self.bart.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            generation_config=self.gen_config
        )
        
        # Decode the generated tokens
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return generated_text

    def load_predict(self, task: str, model: str, text: str):
        self.load_bart(task, model)
        return self.predict(text)

class BertMessageRegressor:
    def __init__(self, device):
        self.bert = None
        self.tokenizer = None
        self.bert_model = None
        self.device = device

    def load_bert(self, task: str, model: str):
        bert_dir = f"./out/models/{task}"
        bert_dir += f"/messages-regression-explanation"
        bert_dir += f"/BERT/"

        mdirs = sorted([mdir for mdir in os.listdir(bert_dir) if model in mdir])
        bert_dir += mdirs[0]

        print(f"Loading BERT model from dir: {bert_dir}")

        last_checkpoint = sorted(
            [f for f in os.listdir(bert_dir) if f.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
            reverse=True
        )[0]
        print(f"Loading BERT model from checkpoint: {last_checkpoint}")

        bert_dir += f"/{last_checkpoint}"

        if self.bert_model != bert_dir:
            self.bert_model = bert_dir
            self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
            self.bert = BertForSequenceClassification.from_pretrained(
                bert_dir,
                num_labels=1, # our task is regression (num_labels=1).
                problem_type="regression"
            ).to(self.device)

    def preprocess(self, msgs, target_idx):
        out = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        }

        target = f"[SEP]{msgs[target_idx]['person']}:\n{msgs[target_idx]['message']}[SEP]"
        prefix = self.tokenizer(
            "[CLS]" + "\n".join([f"{msg['person']}:\n{msg['message']}" for msg in msgs[:target_idx]]),
            max_length=512, truncation=True,
            add_special_tokens=False
        )
        prefix['token_type_ids'][0] = 1

        target = self.tokenizer(
            target,
            max_length=512, truncation=True,
            add_special_tokens=False
        )
        target['input_type_ids'] = [1] * len(target['input_ids'])

        suffix = self.tokenizer(
            "\n".join([f"{msg['person']}:\n{msg['message']}" for msg in msgs[target_idx + 1:]]),
            max_length=512, truncation=True,
            add_special_tokens=False
        )

        out['input_ids'].extend(prefix['input_ids'])
        out['input_ids'].extend(target['input_ids'])
        out['input_ids'].extend(suffix['input_ids'])

        out['attention_mask'].extend(prefix['attention_mask'])
        out['attention_mask'].extend(target['attention_mask'])
        out['attention_mask'].extend(suffix['attention_mask'])

        out['token_type_ids'].extend(prefix['token_type_ids'])
        out['token_type_ids'].extend(target['token_type_ids'])
        out['token_type_ids'].extend(suffix['token_type_ids'])

        print(self.tokenizer.decode(out['input_ids'], skip_special_tokens=False))

        out = {
            'input_ids': torch.tensor(out['input_ids'], device=self.device).unsqueeze(0),
            'attention_mask': torch.tensor(out['attention_mask'], device=self.device).unsqueeze(0),
            'token_type_ids': torch.tensor(out['token_type_ids'], device=self.device).unsqueeze(0)
        }
        return out

    def predict(self, msgs, target_idx):
        if not self.bert or not self.tokenizer:
            raise ValueError("BERT model or tokenizer not loaded")

        inputs = self.preprocess(msgs, target_idx)
        outputs = self.bert(**inputs)
        return outputs.logits

    def load_predict(self, task: str, model: str, msgs: list, target_idx: int):
        self.load_bert(task, model)
        return self.predict(msgs, target_idx)


# class BertChatRegressor:
#     def __init__(self, device):
#         self.bert = None
#         self.tokenizer = None
#         self.bert_model = None
#         self.device = device

#     def load_bert(self, task: str, labels: str, model: str):
#         bert_dir = f"./out/models/{task}"
#         bert_dir += f"/messages-regression-explanation"
#         bert_dir += f"/BERT/latest-{model}"

#         last_checkpoint = sorted(
#             [f for f in os.listdir(bert_dir) if f.startswith("checkpoint-")],
#             key=lambda x: int(x.split("-")[1]),
#             reverse=True
#         )[0]

#         bert_dir += f"/{last_checkpoint}"

#         if self.bert_model != bert_dir:
#             self.bert_model = bert_dir
#             self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
#             self.bert = BertForMultipleRegressions.from_pretrained(
#                 bert_dir, self.tokenizer.cls_token_id
#             ).to(self.device)
#             temp = model.split("-")
#             self.with_token_type_ids = temp[-1].split("_")[-1] == 'True'
#             print(f"with_token_type_ids: {self.with_token_type_ids}")
#             self.with_sep_tokens = temp[-2].split("_")[-1] == 'True'
#             print(f"with_sep_tokens: {self.with_sep_tokens}")

#     def preprocess(self, msgs):
#         out = {
#             'input_ids': [],
#             'attention_mask': [],
#             'token_type_ids': []
#         }
#         for i, msg in enumerate(msgs):
#             # {msg['person']}:
#             chat_part = f"{msg['message']}[CLS]"
#             if self.with_sep_tokens:
#                 chat_part += '[SEP]'
#             tknzd_msg = self.tokenizer(
#                 chat_part, # return_tensors="pt",
#                 max_length=512, truncation=True,
#                 add_special_tokens=False
#             )
#             out['input_ids'].extend(tknzd_msg['input_ids'])
#             out['attention_mask'].extend(tknzd_msg['attention_mask'])
#             if self.with_token_type_ids:
#                 out['token_type_ids'].extend([msg['person_id']] * len(tknzd_msg['input_ids']))
#             else:
#                 out['token_type_ids'].extend(tknzd_msg['token_type_ids'])
#         print(self.tokenizer.decode(out['input_ids'], skip_special_tokens=False))

#         out = {
#             'input_ids': torch.tensor(out['input_ids'], device=self.device).unsqueeze(0),
#             'attention_mask': torch.tensor(out['attention_mask'], device=self.device).unsqueeze(0),
#             'token_type_ids': torch.tensor(out['token_type_ids'], device=self.device).unsqueeze(0)
#         }
#         return out

#     def predict(self, msgs):
#         if not self.bert or not self.tokenizer:
#             raise ValueError("BERT model or tokenizer not loaded")

#         inputs = self.preprocess(msgs)
#         outputs = self.bert(**inputs)
#         return outputs.logits

#     def load_predict(self, task: str, labels: str, model: str, text: str):
#         self.load_bert(task, labels, model)
#         return self.predict(text)

# class BartChatRegressorExplanator:

#     def __init__(self, device):
#         self.bart = None
#         self.tokenizer = None
#         self.bart_model = None
#         self.device = device
#         self.gen_kwargs = {
#             "max_length": 1024,
#             "do_sample": True,
#             "top_p": 0.95,
#             "top_k": 25,
#             "temperature": 0.6,
#         }

#     def load_bart(self, task: str, labels: str, model: str):
#         bart_dir = f"./out/models/{task}"
#         bart_dir += f"/messages-regression-explanation"
#         bart_dir += f"/Multi-Task BART/{model}"

#         print(bart_dir)
#         if self.bart_model != bart_dir:
#             self.bart_model = bart_dir
#             self.bart = BartForConditionalGenerationAndRegression(
#                 load_path=bart_dir,
#                 single_sep_token=False,
#                 verbose=True
#             ).to(self.device)
#             self.tokenizer = BartWithRegression.get_tokenizer(single_sep_token=False)

#     def preprocess(self, msgs):
#         out = {
#             'input_ids': [],
#             'attention_mask': []
#         }
#         for i, msg in enumerate(msgs):
#             # {msg['timestamp']} | 
#             chat_part = f"{msg['person']}:\n{msg['message']}[USR{msg['person_id']}]\n"
#             tknzd_msg = self.tokenizer(
#                 chat_part,
#                 max_length=1024, truncation=True,
#                 add_special_tokens=False
#             )
#             out['input_ids'].extend(tknzd_msg['input_ids'])
#             out['attention_mask'].extend(tknzd_msg['attention_mask'])

#         print(self.tokenizer.decode(out['input_ids'], skip_special_tokens=False))

#         out = {
#             'input_ids': torch.tensor(out['input_ids'], device=self.device).unsqueeze(0),
#             'attention_mask': torch.tensor(out['attention_mask'], device=self.device).unsqueeze(0)
#         }
#         return out

#     def predict(self, msgs):
#         if not self.bart or not self.tokenizer:
#             raise ValueError("BART model or tokenizer not loaded")

#         inputs = self.preprocess(msgs)
#         outputs = self.bart(gen_kwargs=self.gen_kwargs, **inputs)

#         explanation = self.tokenizer.decode(
#             outputs['explanations'][0], skip_special_tokens=True
#         )

#         return {
#             'explanations': explanation,
#             'polarities': outputs['polarities'].tolist()
#         }

#     def load_predict(self, task: str, labels: str, model: str, text: str):
#         self.load_bart(task, labels, model)
#         return self.predict(text)
