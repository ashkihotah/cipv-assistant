from torch.amp import GradScaler, autocast
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
import json
import os
import warnings

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
        flattened_hidden_states = encoder_last_hidden_state.view(-1, encoder_last_hidden_state.size(-1))
        flattened_sep_mask = sep_token_matrix_mask.view(-1)

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
            'generation_loss': outputs.loss
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
        
        # Handle DDP wrapped models
        model_to_save = self.module if hasattr(self, 'module') else self
        model_to_save.bart.save_pretrained(save_directory)
        torch.save(model_to_save.regression_head.state_dict(), os.path.join(save_directory, "regression_head.pt"))

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

def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def get_device_info():
    """Get device information and setup strategy"""
    if not torch.cuda.is_available():
        return 'cpu', 1, 'single'
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        return 'cuda:0', 1, 'single'
    elif num_gpus > 1:
        return 'cuda', num_gpus, 'parallel'
    
    return 'cpu', 1, 'single'

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
            # Multi-GPU specific arguments
            use_distributed: bool=False,
            local_rank: int=-1,
            find_unused_parameters: bool=False
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
        # Multi-GPU arguments
        self.use_distributed = use_distributed
        self.local_rank = local_rank
        self.find_unused_parameters = find_unused_parameters

class Trainer:
    """
    A simple trainer class to handle the training loop for the hybrid model.
    It abstracts away the training logic, allowing for easy integration with
    different datasets and models with multi-GPU support.
    """
    def __init__(
            self, model: BartWithRegression, device,
            args: TrainingArguments=None,
            train_dataloader=None, eval_dataloader=None, test_dataloader=None,
            eval_sts_model='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        ):
        self.original_model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.args = args
        
        # Device and multi-GPU setup
        device_name, self.num_gpus, self.strategy = get_device_info()
        self.is_distributed = args.use_distributed if args else False
        self.local_rank = args.local_rank if args else -1
        
        # Setup model for multi-GPU training
        self.model = self._setup_model_for_training()
        
        # Move model to appropriate device
        if self.strategy == 'single':
            self.model = self.model.to(self.device)
        elif self.strategy == 'parallel' and not self.is_distributed:
            self.model = self.model.to(self.device)
            if self.num_gpus > 1:
                print(f"Using DataParallel with {self.num_gpus} GPUs")
                self.model = DataParallel(self.model)

        if self.args is not None:
            self._setup_optimizer_and_scheduler()
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

        # Setup evaluation model (only on main process for distributed training)
        if not self.is_distributed or self.local_rank <= 0:
            print(f"Loading Evaluation STS model: {eval_sts_model}")
            self.sts_model = SentenceTransformer(eval_sts_model, device=self.device)
        else:
            self.sts_model = None

    def _setup_model_for_training(self):
        """Setup model for different training strategies"""
        model = self.original_model
        
        if self.is_distributed:
            # For distributed training, wrap with DDP
            if self.local_rank >= 0:
                model = model.to(f'cuda:{self.local_rank}')
                model = DDP(
                    model, 
                    device_ids=[self.local_rank],
                    find_unused_parameters=self.args.find_unused_parameters if self.args else False
                )
                print(f"Using DistributedDataParallel on rank {self.local_rank}")
        
        return model

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Get the actual model (unwrap if necessary)
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Define parameter groups for the optimizer
        bart_params = actual_model.bart.parameters()
        regression_head_params = actual_model.regression_head.parameters()
        optimizer_grouped_parameters = [
            {"params": bart_params, "lr": self.args.body_lr},
            {"params": regression_head_params, "lr": self.args.head_lr}
        ]

        # Add UncertaintyLoss parameters if applicable
        if isinstance(self.args.criterion, UncertaintyLoss):
            uncertainty_loss_params = self.args.criterion.parameters()
            optimizer_grouped_parameters.append(
                {"params": uncertainty_loss_params, "lr": self.args.head_lr}
            )

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            weight_decay=self.args.weight_decay
        )

        # Adjust number of training steps for distributed training
        num_training_steps = len(self.train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_epochs
        if self.is_distributed:
            # In distributed training, each process sees only a portion of the data
            num_training_steps = num_training_steps // self.num_gpus
        
        num_warmup_steps = int(num_training_steps * self.args.warmup_percentage)
        
        self.lr_scheduler = self.args.get_scheduler_fn(
            self.optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def _should_log_and_save(self):
        """Determine if current process should log and save (only main process in distributed)"""
        return not self.is_distributed or self.local_rank <= 0

    def validate(self):
        """
        Validates the model on the provided dataloader using the specified
        criterion. Returns the average loss over the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        reg_loss = 0.0
        gen_loss = 0.0
        
        # Only show progress bar on main process
        show_progress = self._should_log_and_save()
        
        with torch.no_grad():
            eval_progress_bar = tqdm(self.eval_dataloader, desc="Validation", disable=not show_progress)
            for batch in eval_progress_bar:
                true_polarities = batch.pop('polarities').detach().clone()
                
                # Move to appropriate device
                if self.is_distributed:
                    true_polarities = true_polarities.to(f'cuda:{self.local_rank}')
                    for k in batch: 
                        batch[k] = batch[k].to(f'cuda:{self.local_rank}')
                else:
                    true_polarities = true_polarities.to(self.device)
                    for k in batch: 
                        batch[k] = batch[k].to(self.device)

                device_type = 'cuda' if str(self.device).startswith('cuda') else 'cpu'
                with autocast(device_type=device_type):
                    model_outputs = self.model(**batch)
                    loss = self.args.criterion(model_outputs, true_polarities)

                total_loss += loss['total_loss']
                reg_loss += loss['reg_loss']
                gen_loss += loss['gen_loss']

                if show_progress:
                    avg_total_loss = total_loss / len(self.eval_dataloader)
                    avg_reg_loss = reg_loss / len(self.eval_dataloader)
                    avg_gen_loss = gen_loss / len(self.eval_dataloader)

                    eval_progress_bar.set_postfix({
                        'avg_total_loss': avg_total_loss.item(),
                        'avg_reg_loss': avg_reg_loss.item(),
                        'avg_gen_loss': avg_gen_loss.item()
                    })

        # Synchronize losses across all processes in distributed training
        if self.is_distributed:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(reg_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(gen_loss, op=dist.ReduceOp.SUM)
            total_loss /= self.num_gpus
            reg_loss /= self.num_gpus
            gen_loss /= self.num_gpus

        avg_total_loss = total_loss / len(self.eval_dataloader)
        avg_reg_loss = reg_loss / len(self.eval_dataloader)
        avg_gen_loss = gen_loss / len(self.eval_dataloader)

        return {
            'total_loss': avg_total_loss,
            'reg_loss': avg_reg_loss,
            'gen_loss': avg_gen_loss
        }

    def train(self, trial: optuna.Trial = None):
        if self.args is None:
            raise ValueError("Training arguments must be provided!")
    
        if self.train_dataloader is None:
            raise ValueError("Training dataloader must be provided!")

        if not isinstance(self.args.criterion, BartWithRegressionCriterion):
            raise TypeError("Criterion must be an instance of BartWithRegressionCriterion!")

        # Fundamental if using mixed precision training
        scaler = GradScaler()
        best_eval_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.args.num_epochs):
            self.model.train()
            
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            total_train_loss = 0.0
            show_progress = self._should_log_and_save()
            
            train_progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}",
                disable=not show_progress
            )

            for batch_idx, batch in enumerate(train_progress_bar):
                # Pop polarity labels from batch
                true_polarities = batch.pop('polarities').detach().clone()

                # Move all batch tensors to appropriate device
                if self.is_distributed:
                    true_polarities = true_polarities.to(f'cuda:{self.local_rank}')
                    for k in batch: batch[k] = batch[k].to(f'cuda:{self.local_rank}')
                else:
                    true_polarities = true_polarities.to(self.device)
                    for k in batch: batch[k] = batch[k].to(self.device)

                # Use autocast for mixed precision training
                device_type = 'cuda' if str(self.device).startswith('cuda') else 'cpu'
                with autocast(device_type=device_type):
                    model_outputs = self.model(**batch)
                    losses_dict = self.args.criterion(
                        model_outputs=model_outputs,
                        true_polarities=true_polarities
                    )

                # Reduce loss to a scalar
                total_train_loss += losses_dict['total_loss'].item()

                # Scale the loss and call backward
                normalized_loss = losses_dict['total_loss'] / self.args.gradient_accumulation_steps
                scaler.scale(normalized_loss).backward()

                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # Unscale gradients and call optimizer.step()
                    scaler.step(self.optimizer)
                    scaler.update()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Update progress bar (only on main process)
                if show_progress:
                    train_progress_bar.set_postfix({
                        'avg loss': total_train_loss / (batch_idx + 1),
                        'reg_loss': losses_dict['reg_loss'].item(),
                        'gen_loss': losses_dict['gen_loss'].item(),
                        'lr_body': self.optimizer.param_groups[0]['lr'],
                        'lr_head': self.optimizer.param_groups[1]['lr'],
                    })

            avg_train_loss = total_train_loss / len(self.train_dataloader)

            # Validation and early stopping (only on main process)
            eval_loss = None
            if self.eval_dataloader is not None:
                eval_losses_dict = self.validate()
                eval_loss = eval_losses_dict['total_loss'].item()
                
                if self._should_log_and_save():
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        patience_counter = 0
                        if self.args.save_path is not None:
                            # Get the actual model (unwrap if necessary)
                            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                            model_to_save.save_model(self.args.save_path)
                    else:
                        if self.args.early_stopping_patience is not None:
                            patience_counter += 1
                            if patience_counter >= self.args.early_stopping_patience:
                                print(f"Early stopping triggered after {epoch+1} epochs")
                                return

            # Logging (only on main process)
            if self.args.logging and self._should_log_and_save():
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
            
            # Optuna Pruning Integration (only on main process)
            if trial is not None and eval_loss is not None and self._should_log_and_save():
                trial.report(eval_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    def evaluate(self, leave=True, **gen_kwargs):
        # Only evaluate on main process in distributed training
        if self.is_distributed and self.local_rank > 0:
            return {}
            
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
                true_polarities = batch.pop('polarities').detach().clone()
                true_explanation_ids = batch.pop('labels').detach().clone()
                
                # Move to appropriate device
                if self.is_distributed:
                    true_polarities = true_polarities.to(f'cuda:{self.local_rank}')
                    true_explanation_ids = true_explanation_ids.to(f'cuda:{self.local_rank}')
                    for k in batch: batch[k] = batch[k].to(f'cuda:{self.local_rank}')
                else:
                    true_polarities = true_polarities.to(self.device)
                    true_explanation_ids = true_explanation_ids.to(self.device)
                    for k in batch: batch[k] = batch[k].to(self.device)

                model_outputs = self.model.generate(**batch, **gen_kwargs)

                all_polarities_pred.extend(model_outputs['polarities'].tolist())
                all_polarities_true.extend(true_polarities.cpu().tolist())

                # Get tokenizer from the actual model
                tokenizer = (self.model.module if hasattr(self.model, 'module') else self.model).tokenizer

                predicted_explanations = tokenizer.batch_decode(
                    model_outputs['explanations'],
                    skip_special_tokens=True
                )
                true_explanation_ids = torch.where(
                    true_explanation_ids == -100,
                    torch.tensor(tokenizer.pad_token_id).to(true_explanation_ids.device),
                    true_explanation_ids
                )
                true_explanations = tokenizer.batch_decode(
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

        # Calculate similarity scores
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

