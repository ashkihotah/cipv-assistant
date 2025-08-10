from torch.amp import GradScaler, autocast
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import (
    BartConfig,
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer, util
import evaluate as ev
from tqdm.auto import tqdm
import os
import gc

import optuna

from multi_task_bart.losses import (
    BartWithRegressionCriterion,
    UncertaintyLoss
)

class BartWithRegression(nn.Module):
    """
    A hybrid model combining BART for explanation generation and a custom
    regression head for polarity prediction.
    """

    DEFAULT_CHECKPOINT = "morenolq/bart-it"
    SEP_TOKEN = "[SEP]"
    USR0_TOKEN = "[USR0]"
    USR1_TOKEN = "[USR1]"

    def __initialize_architecture(self, regression_dropout: float):
        """
        Initializes the architecture of the model, including the BART model
        and the regression head.
        """
        self.regression_head = nn.Sequential(
            nn.Dropout(regression_dropout),
            nn.Linear(self.bart.config.hidden_size, 1),
            nn.Tanh()
        )

    def __init__(
            self,
            load_path: str=None,
            single_sep_token: bool=False,
            init_sep_tokens: str = None,
            regression_dropout: float=0.1,
            verbose: bool=False
    ):
        super().__init__()
        self.single_sep_token = single_sep_token
        self.tokenizer = BartWithRegression.get_tokenizer(single_sep_token)
        if load_path != None and os.path.exists(load_path):
            self.bart = BartForConditionalGeneration.from_pretrained(load_path)
            self.__initialize_architecture(regression_dropout)
            if "regression_head.pt" in os.listdir(load_path):
                self.regression_head.load_state_dict(
                    torch.load(
                        os.path.join(load_path, "regression_head.pt"),
                        map_location=torch.device('cpu')
                    )
                )
            else:
                raise FileNotFoundError(
                    f"'regression_head.pt' not found in {load_path}."
                )
        else:
            self.bart = BartForConditionalGeneration.from_pretrained(self.DEFAULT_CHECKPOINT)
            
            self.bart.resize_token_embeddings(len(self.tokenizer))
            if self.single_sep_token:
                self.bart.config.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP_TOKEN)

                if init_sep_tokens == "mean":
                    sep_id = self.bart.config.sep_token_id
                    sep_open_id = self.tokenizer.convert_tokens_to_ids("<s>")
                    sep_close_id = self.tokenizer.convert_tokens_to_ids("</s>")
                    with torch.no_grad():
                        emb = self.bart.model.shared
                        mean_emb = (emb.weight[sep_open_id] + emb.weight[sep_close_id]) / 2
                        emb.weight[sep_id] = mean_emb
                elif init_sep_tokens == "</s>":
                    sep_id = self.bart.config.sep_token_id
                    sep_open_id = self.tokenizer.convert_tokens_to_ids("</s>")
                    with torch.no_grad():
                        emb = self.bart.model.shared
                        mean_emb = (emb.weight[sep_open_id] + emb.weight[sep_close_id]) / 2
                        emb.weight[sep_id] = mean_emb
            else:
                self.bart.config.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.USR0_TOKEN)
                self.bart.config.sep_token_id2 = self.tokenizer.convert_tokens_to_ids(self.USR1_TOKEN)

            self.__initialize_architecture(regression_dropout)
            if verbose:
                print("[WARNING]: Local model checkpoint not found.")
                print("It will be initialized with default configurations!")

    def __predict_polarities(
            self,
            encoder_last_hidden_state: torch.Tensor,
            sep_token_matrix_mask: torch.Tensor
        ) -> torch.Tensor:
        """
        Predicts the polarities for specific token indices in the encoder's
        last hidden state.
        """
        batch_size = encoder_last_hidden_state.size(0)
        # print(f"encoder_last_hidden_state.shape: {encoder_last_hidden_state.shape}")
        # print(f"sep_token_matrix_mask.shape: {sep_token_matrix_mask.shape}")
        flattened_hidden_states = encoder_last_hidden_state.view(-1, encoder_last_hidden_state.size(-1))
        flattened_sep_mask = sep_token_matrix_mask.view(-1)

        # print(f"flattened_hidden_states.shape: {flattened_hidden_states.shape}")
        # print(f"flattened_sep_mask.shape: {flattened_sep_mask.shape}")

        predicted_polarities = self.regression_head(flattened_hidden_states[flattened_sep_mask]).squeeze(-1)

        return predicted_polarities

    def forward(
        self,
        input_ids,
        attention_mask,
        labels
    ):
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            # output_attentions=False,
        )

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        sep_token_matrix_mask = (
            input_ids == self.bart.config.sep_token_id
        )
        if not self.single_sep_token:
            sep_token_matrix_mask = sep_token_matrix_mask | (
                (input_ids == self.bart.config.sep_token_id2)
            )
        predicted_polarities = self.__predict_polarities(encoder_last_hidden_state, sep_token_matrix_mask)

        return {
            'polarities': predicted_polarities,
            'explanation_logits': outputs.logits,
            'generation_loss': outputs.loss  # BART's pre-computed generation loss
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **gen_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generates both explanations and polarities from the input data.
        The explanations are generated using the BART model, and the polarities
        are predicted using the regression head.
        """
        # Get encoder outputs
        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        encoder_last_hidden_state = encoder_outputs.last_hidden_state

        # Generate explanations using precomputed encoder outputs
        generated_sequences = self.bart.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            **gen_kwargs
        )

        # Find target token index for polarity prediction
        sep_token_matrix_mask = (
            input_ids == self.bart.config.sep_token_id
        )
        if not self.single_sep_token:
            sep_token_matrix_mask = sep_token_matrix_mask | (
                (input_ids == self.bart.config.sep_token_id2)
            )
        predicted_polarities = self.__predict_polarities(encoder_last_hidden_state, sep_token_matrix_mask)

        return {
            'explanations': generated_sequences,
            'polarities': predicted_polarities
        }

    def save_model(self, save_directory: str):
        """
        Saves the fine-tuned BART model, the regression head, and the tokenizer
        to a specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.bart.save_pretrained(save_directory)
        torch.save(self.regression_head.state_dict(), os.path.join(save_directory, "regression_head.pt"))

    @classmethod
    def get_tokenizer(cls, single_sep_token) -> BartTokenizer:
        tokenizer = BartTokenizer.from_pretrained(cls.DEFAULT_CHECKPOINT)
        if single_sep_token:
            tokenizer.add_special_tokens({
                "additional_special_tokens": [cls.SEP_TOKEN]
            })
        else:
            tokenizer.add_special_tokens({
                "additional_special_tokens": [cls.USR0_TOKEN, cls.USR1_TOKEN]
            })
        return tokenizer

@dataclass
class MultiTaskBartDataCollator(DataCollatorForSeq2Seq):
    """
    Extends DataCollatorForSeq2Seq to also handle custom 'regression_labels'.
    This class pads the input_ids, attention_mask, and labels as usual,
    but also pads the 'regression_labels' to the maximum length in the batch.
    """
    def __call__(
            self,
            features: List[Dict[str, Any]],
            return_tensors: Optional[str]=None
        ) -> Dict[str, torch.Tensor]:
        # Extract polarities from features and create a 1-D tensor
        # containing all polarities concatenated
        polarities = [polarity for feature in features for polarity in feature.pop('polarities')]

        # Let the parent class handle the standard seq2seq padding and
        # creation of decoder_input_ids.
        # This will correctly pad input_ids, attention_mask, and labels.
        batch = super().__call__(features, return_tensors)
        
        # Convert polarities to a tensor
        batch['polarities'] = torch.tensor(polarities, dtype=torch.float32)
        return batch

class TrainingArguments:
    """
    A simple class to hold training arguments for the hybrid model.
    This is useful for passing around training configurations without
    needing to use a dictionary.
    """
    def __init__(
            self, criterion: BartWithRegressionCriterion,
            num_epochs: int=1, gradient_accumulation_steps: int=1,
            get_scheduler_fn=get_linear_schedule_with_warmup,
            warmup_percentage: float=0.1,
            body_lr: float=3e-5, head_lr: float=1.5e-4,
            weight_decay: float=0.01,
            early_stopping_patience: int=None,
            logging: bool=True, save_path: str=None,
            load_best_model_at_end: bool=False
        ):
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.logging = logging
        self.save_path = save_path
        self.body_lr = body_lr
        self.head_lr = head_lr
        self.warmup_percentage = warmup_percentage
        self.weight_decay = weight_decay
        self.get_scheduler_fn = get_scheduler_fn
        self.criterion = criterion
        self.load_best_model_at_end = load_best_model_at_end

class Trainer:
    """
    A simple trainer class to handle the training loop for the hybrid model.
    It abstracts away the training logic, allowing for easy integration with
    different datasets and models.
    """
    def __init__(
            self, model: BartWithRegression, device,
            args: TrainingArguments=None,
            train_dataloader=None, eval_dataloader=None, test_dataloader=None,
            eval_sts_model='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        ):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.args = args

        if self.args is not None:
            # Define parameter groups for the optimizer
            # A common starting point is a 5x to 10x difference in LR
            bart_params = self.model.bart.parameters()
            regression_head_params = self.model.regression_head.parameters()
            optimizer_grouped_parameters = [
                {"params": bart_params, "lr": self.args.body_lr},
                {"params": regression_head_params, "lr": self.args.head_lr}
            ]

            # Add UncertaintyLoss parameters if applicable
            if isinstance(self.args.criterion, UncertaintyLoss):
                uncertainty_loss_params = self.args.criterion.parameters()
                optimizer_grouped_parameters.append(
                    {"params": uncertainty_loss_params, "lr": self.args.head_lr}  # Use head_lr or a custom LR
                )

            # The AdamW optimizer handles the groups seamlessly
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                # It discourages the model from developing very large weights
                # forcing it to use all of its weights to a small extent
                # rather than relying heavily on a few
                weight_decay=self.args.weight_decay
            )

            num_training_steps = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_epochs
            # 10 : 100 = num_warmup_steps : num_training_steps
            num_warmup_steps = int(num_training_steps * self.args.warmup_percentage)
            # The learning rate scheduler will also correctly apply its schedule to each group's base LR
            self.lr_scheduler = self.args.get_scheduler_fn(
                self.optimizer, num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )

            # print(f"Number of training steps: {num_training_steps}")
            # print(f"Number of warmup steps: {num_warmup_steps}")

            self.log_history = {
                'epochs': [],
                'total_train_losses': [],
                'reg_train_losses': [],
                'gen_train_losses': [],
                'total_eval_losses': [],
                'reg_eval_losses': [],
                'gen_eval_losses': [],
                'reg_loss_weight': [],
                'gen_loss_weight': []
            }

        # print(f"Loading Evaluation STS model: {eval_sts_model}")
        self.sts_model = SentenceTransformer(eval_sts_model, device=self.device)

    def validate(self):
        """
        Validates the model on the provided dataloader using the specified
        criterion. Returns the average loss over the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        reg_loss = 0.0
        gen_loss = 0.0
        with torch.no_grad():
            eval_progress_bar = tqdm(self.eval_dataloader, desc="Validation")
            for batch in eval_progress_bar:
                true_polarities = batch.pop('polarities').detach().clone().to(self.device)
                for k in batch: batch[k] = batch[k].to(self.device)

                with autocast(device_type=self.device.type):
                    model_outputs = self.model(**batch)
                    loss = self.args.criterion(model_outputs, true_polarities)

                total_loss += loss['total_loss']
                reg_loss += loss['reg_loss']
                gen_loss += loss['gen_loss']

                avg_total_loss = total_loss / len(self.eval_dataloader)
                avg_reg_loss = reg_loss / len(self.eval_dataloader)
                avg_gen_loss = gen_loss / len(self.eval_dataloader)

                eval_progress_bar.set_postfix({
                    'avg_total_loss': avg_total_loss.item(),
                    'avg_reg_loss': avg_reg_loss.item(),
                    'avg_gen_loss': avg_gen_loss.item()
                })
        return {
            'total_loss': avg_total_loss,
            'reg_loss': avg_reg_loss,
            'gen_loss': avg_gen_loss
        }

    def train(
            self,
            trial: optuna.Trial = None
        ):

        if self.args is None:
            raise ValueError("Training arguments must be provided!")
    
        if self.train_dataloader is None:
            raise ValueError("Training dataloader must be provided!")

        if not isinstance(self.args.criterion, BartWithRegressionCriterion):
            raise TypeError("Criterion must be an instance of BartWithRegressionCriterion!")

        # Fundamental if using mixed precision training
        # See pytorch docs: https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling
        scaler = GradScaler()
        best_eval_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.args.num_epochs):
            self.model.train()
            total_train_loss = 0.0
            train_progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")

            for batch_idx, batch in enumerate(train_progress_bar):

                # Pop polarity and explanation labels from batch
                true_polarities = batch.pop('polarities').detach().clone().to(self.device)
                # true_explanation_ids = torch.tensor(batch.pop('labels')).to(device)

                # Move all batch tensors to device
                for k in batch: batch[k] = batch[k].to(self.device)

                # Use autocast for mixed precision training
                with autocast(device_type=self.device.type):
                    # The model forward can now accept the whole batch, as the DataCollator has prepared it perfectly
                    model_outputs = self.model(**batch)
                    
                    # The criterion now only needs the true polarity, as gen_loss is in the model_outputs
                    losses_dict = self.args.criterion(
                        model_outputs=model_outputs,
                        true_polarities=true_polarities
                    )
                    if not self.model.freeze_bart:
                        loss = losses_dict['total_loss']
                    else:
                        loss = losses_dict['reg_loss']

                # Reduce loss to a scalar
                total_train_loss += loss.item()

                # Scale the loss and call backward
                normalized_loss = loss / self.args.gradient_accumulation_steps # Normalize loss
                scaler.scale(normalized_loss).backward()

                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # Unscale gradients and call optimizer.step()
                    scaler.step(self.optimizer)
                    # Update the scale for next iteration
                    scaler.update()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step() # Update learning rate
                    self.optimizer.zero_grad()

                # Update progress bar
                train_progress_bar.set_postfix({
                    'avg loss': total_train_loss / (batch_idx + 1),
                    'reg_loss': losses_dict['reg_loss'].item(),
                    'gen_loss': losses_dict['gen_loss'].item(),
                    'lr_body': self.optimizer.param_groups[0]['lr'],
                    'lr_head': self.optimizer.param_groups[1]['lr'],
                    # 'reg_weight': self.args.criterion.log_var_polarity.item(),
                    # 'gen_weight': self.args.criterion.log_var_explanation.item()
                })

            avg_train_loss = total_train_loss / len(self.train_dataloader)

            if self.eval_dataloader is not None:
                eval_losses_dict = self.validate()
                eval_loss = eval_losses_dict['total_loss'].item()
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                    if self.args.save_path is not None:
                        self.model.save_model(os.path.join(self.args.save_path))
                else:
                    if self.args.early_stopping_patience is not None:
                        patience_counter += 1
                        if patience_counter >= self.args.early_stopping_patience:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                            return

            if self.args.logging:
                self.log_history['epochs'].append(epoch + 1)
                self.log_history['total_train_losses'].append(avg_train_loss)
                self.log_history['reg_train_losses'].append(losses_dict['reg_loss'].item())
                self.log_history['gen_train_losses'].append(losses_dict['gen_loss'].item())
                if isinstance(self.args.criterion, UncertaintyLoss):
                    self.log_history['reg_loss_weight'].append(self.args.criterion.log_var_polarity.item())
                    self.log_history['gen_loss_weight'].append(self.args.criterion.log_var_explanation.item())
                if self.eval_dataloader is not None:
                    self.log_history['total_eval_losses'].append(eval_losses_dict['total_loss'].item())
                    self.log_history['reg_eval_losses'].append(eval_losses_dict['reg_loss'].item())
                    self.log_history['gen_eval_losses'].append(eval_losses_dict['gen_loss'].item())
            
            # --- NEW: Optuna Pruning Integration ---
            if trial is not None:
                # Report the intermediate evaluation loss to Optuna
                trial.report(eval_loss, epoch)
                # Check if the trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            # --- END NEW ---

        if self.args.load_best_model_at_end and self.args.save_path is not None:
            del self.model
            gc.collect()

            self.model = BartWithRegression.load_model(
                self.args.save_path,
                single_sep_token=self.model.single_sep_token,
                regression_dropout=self.model.regression_head[0].p
            )

    def evaluate(self, leave=True, **gen_kwargs):
        self.model.eval()

        rouge_metric = ev.load('rouge')
        bleu_metric = ev.load('bleu')
        bertscore_metric = ev.load('bertscore')
        all_polarities_pred = []
        all_polarities_true = []
        all_true_explanations = []
        all_predicted_explanations = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Evaluating", leave=leave):
                true_polarities = batch.pop('polarities').detach().clone().to(self.device)
                true_explanation_ids = batch.pop('labels').detach().clone().to(self.device)
                for k in batch: batch[k] = batch[k].to(self.device)

                model_outputs = self.model.generate(**batch, **gen_kwargs)

                all_polarities_pred.extend(model_outputs['polarities'].tolist())
                all_polarities_true.extend(true_polarities)

                predicted_explanations = self.model.tokenizer.batch_decode(
                    model_outputs['explanations'],
                    skip_special_tokens=True
                )
                true_explanation_ids = torch.where(
                    true_explanation_ids == -100,
                    torch.tensor(self.model.tokenizer.pad_token_id)
                        .to(true_explanation_ids.device),
                    true_explanation_ids
                )
                true_explanations = self.model.tokenizer.batch_decode(
                    true_explanation_ids,
                    skip_special_tokens=True
                )
                
                all_true_explanations.extend(true_explanations)
                all_predicted_explanations.extend(predicted_explanations)

                rouge_metric.add_batch(
                    predictions=predicted_explanations,
                    references=true_explanations
                )
                bleu_metric.add_batch(
                    predictions=predicted_explanations,
                    references=[[ref] for ref in true_explanations]
                )

        reference_embeddings = self.sts_model.encode(all_true_explanations, convert_to_tensor=True)
        generated_embeddings = self.sts_model.encode(all_predicted_explanations, convert_to_tensor=True)
        cosine_scores = util.cos_sim(generated_embeddings, reference_embeddings)
        sbert_similarity = torch.diag(cosine_scores).mean().item()

        bertscore_results = bertscore_metric.compute(
            predictions=all_predicted_explanations, 
            references=all_true_explanations, 
            lang="it",
            device=self.device
        )
        bertscore_f1 = sum(bertscore_results['f1']) / len(bertscore_results['f1'])

        rouge_score = rouge_metric.compute()
        bleu_score = bleu_metric.compute()
        mse = F.mse_loss(
            torch.tensor(all_polarities_pred),
            torch.tensor(all_polarities_true)
        ).item()

        return {
            'rouge1': rouge_score['rouge1'],
            'rouge2': rouge_score['rouge2'],
            'rougeL': rouge_score['rougeL'],
            'bleu': bleu_score['bleu'],
            'mse': mse,
            'sbert_similarity': sbert_similarity,
            'bertscore_f1': bertscore_f1
        }
