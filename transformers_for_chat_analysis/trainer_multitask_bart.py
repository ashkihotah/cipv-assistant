
@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    """
    A simple class to hold training arguments for the hybrid model.
    This is useful for passing around training configurations without
    needing to use a dictionary.
    """
    def __init__(
            self,
            criterion: BartWithRegressionCriterion,
            num_epochs: int=1,
            gradient_accumulation_steps: int=1,
            get_scheduler_fn=get_linear_schedule_with_warmup,
            warmup_percentage: float=0.1,
            body_lr: float=3e-5, 
            head_lr: float=1.5e-4,
            weight_decay: float=0.01,
            early_stopping_patience: int=None,
            logging: bool=True, 
            save_path: str=None,
            load_best_model_at_end: bool=False,
            gen_kwargs: Dict[str, Any]=None,
            # predict_with_generation: bool=False,
            freeze_bart_body: bool=False,
            metric_for_best_model: str='eval_combined_loss',
            greater_is_better: bool=False
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
        self.freeze_bart_body = freeze_bart_body
        self.gen_kwargs = gen_kwargs
        # self.predict_with_generation = predict_with_generation
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better

class Trainer:
    """
    A simple trainer class to handle the training loop for the hybrid model.
    It abstracts away the training logic, allowing for easy integration with
    different datasets and models.
    """
    # freeze e lr diversi
    # nel log appaiono combined/reg/gen loss, i due lr

    def __freeze_bart_body(self):
        """
        Freezes the body of the BART model, allowing only the regression head
        and new embeddings to be trained. This is useful for transfer learning
        scenarios.
        """
        for param in self.model.bart.parameters():
            param.requires_grad = False

        # Unfreeze the shared embedding layer and final bias term if they exist.
        self.model.bart.model.shared.requires_grad_(True)
        if hasattr(self.model.bart, 'final_logits_bias'):
            self.model.bart.final_logits_bias.requires_grad_(True)

    def __create_optimizer(self):
        """
        Creates the optimizer and learning rate scheduler for the model.
        The optimizer is configured to handle different learning rates for
        the BART body and the regression head.
        """
        bart_params = filter(lambda p: p.requires_grad, self.model.bart.parameters())
        regression_head_params = self.model.regression_head.parameters()
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

        # The AdamW optimizer handles the groups seamlessly
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            weight_decay=self.args.weight_decay
        )

    def __create_scheduler(self):
        """
        Creates the learning rate scheduler for the optimizer. The scheduler
        is configured to handle warmup steps based on the total number of
        training steps and the specified warmup percentage.
        """
        num_training_steps = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_epochs
        num_warmup_steps = int(num_training_steps * self.args.warmup_percentage)
        self.lr_scheduler = self.args.get_scheduler_fn(
            self.optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def __init__(
            self, 
            model: BartForConditionalGenerationAndRegression, 
            device,
            args: TrainingArguments,
            train_dataloader, 
            eval_dataloader, 
            test_dataloader,
            compute_metrics=None
        ):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.args = args
        self.compute_metrics = compute_metrics

        # Convert the list of new token IDs to a tensor for efficient use on the GPU.
        self.new_token_ids_tensor = torch.tensor(
            self.model.new_token_ids, 
            dtype=torch.long
        ).to(self.device)

        self.scaler = GradScaler()
        self.patience_counter = 0

        if self.args.freeze_bart_body:
            self.__freeze_bart_body()

        self.__create_optimizer()
        self.__create_scheduler()
        
        self.log_history = []

    def evaluate(self, prefix: str="eval_"):
        """
        Validates the model on the provided dataloader using the specified
        criterion. Returns the average loss over the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        reg_loss = 0.0
        gen_loss = 0.0

        all_preds = []
        all_labels = []
        # all_polarities_pred = []
        # all_polarities_true = []

        with torch.no_grad():
            eval_progress_bar = tqdm(self.eval_dataloader, desc="Validation", leave=False, position=1)
            for batch in eval_progress_bar:
                true_polarities = batch.pop('regression_labels').detach().clone().to(self.device)
                # Keep a copy of labels for metric calculation
                labels = batch['labels'].detach().clone()
                for k in batch: batch[k] = batch[k].to(self.device)

                with autocast(device_type=self.device.type):
                    model_outputs = self.model(**batch, gen_kwargs=self.args.gen_kwargs)
                    loss = self.args.criterion(model_outputs, true_polarities)

                total_loss += loss['total_loss'].item()
                reg_loss += loss['reg_loss'].item()
                gen_loss += loss['gen_loss'].item()

                if self.compute_metrics is not None:
                    # all_polarities_pred.extend(model_outputs['polarities'].cpu().tolist())
                    # all_polarities_true.extend(true_polarities.cpu().tolist())
                    
                    if self.args.gen_kwargs is not None:
                        all_preds.extend(model_outputs['explanations'])
                        all_labels.extend(labels)

            loss_dict = {
                prefix + 'combined_loss': total_loss / len(self.eval_dataloader),
                prefix + 'reg_loss': reg_loss / len(self.eval_dataloader),
                prefix + 'gen_loss': gen_loss / len(self.eval_dataloader)
            }

            metrics_dict = {}
            if self.compute_metrics is not None:
                # eval_preds = {
                #     "predictions": all_preds,
                #     "references": all_labels,
                #     "polarities_pred": all_polarities_pred,
                #     "polarities_true": all_polarities_true
                # }
                all_preds = torch.nn.utils.rnn.pad_sequence(
                    all_preds, batch_first=True,
                    padding_value=self.model.tokenizer.pad_token_id
                )
                all_labels = torch.nn.utils.rnn.pad_sequence(
                    all_labels, batch_first=True,
                    padding_value=self.model.tokenizer.pad_token_id
                )
                metrics_dict = self.compute_metrics((all_preds, all_labels))
                metrics_dict = {f"{prefix}{k}": v for k, v in metrics_dict.items()}
            
        return {**loss_dict, **metrics_dict}

    def __epoch_step(self):
        self.model.train()
        total_train_loss = 0.0

        # epoch_bar = tqdm(self.train_dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(self.train_dataloader):
            true_polarities = batch.pop(
                'polarities'
            ).detach().clone().to(self.device)
            for k in batch: batch[k] = batch[k].to(self.device)

            with autocast(device_type=self.device.type):
                model_outputs = self.model(**batch)
                losses_dict = self.args.criterion(
                    model_outputs=model_outputs,
                    true_polarities=true_polarities
                )
                loss = losses_dict['total_loss']

            total_train_loss += loss.item()
            normalized_loss = loss / self.args.gradient_accumulation_steps
            self.scaler.scale(normalized_loss).backward()

            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                # if self.model.bart.model.shared.weight.grad is not None:
                # Calculate the scaling factor for the new embedding gradients.
                lr_scale = self.args.head_lr / self.args.body_lr
                
                # Create a mask of ones with the same shape as the gradients.
                grad_mask = torch.ones_like(self.model.bart.model.shared.weight.grad)
                
                # Set the mask to the scaling factor only at the indices of the new tokens.
                grad_mask.index_fill_(0, self.new_token_ids_tensor, lr_scale)
                
                # Apply the mask to the gradients.
                self.model.bart.model.shared.weight.grad.mul_(grad_mask)

                if self.args.freeze_bart_body:
                    # if self.model.bart.model.shared.weight.grad is not None:
                    # Create a mask of zeros with the same shape as the gradients.
                    grad_mask = torch.zeros_like(self.model.bart.model.shared.weight.grad)
                    # Set the mask to 1.0 only at the indices of the new tokens.
                    grad_mask.index_fill_(0, self.new_token_ids_tensor, 1.0)
                    # Apply the mask to the gradients.
                    self.model.bart.model.shared.weight.grad.mul_(grad_mask)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step() # Update learning rate
                self.optimizer.zero_grad()

            self.epochs_bar.set_postfix({
                'lr_body': self.optimizer.param_groups[0]['lr'],
                'lr_head': self.optimizer.param_groups[1]['lr'],
            })

        avg_train_loss = total_train_loss / len(self.train_dataloader)

        self.log_history.append({
            'train_combined_loss': avg_train_loss,
            'train_reg_loss': losses_dict['reg_loss'].item(),
            'train_gen_loss': losses_dict['gen_loss'].item(),
        })

    def __validate_step(self):
        eval_losses_dict = self.evaluate(prefix="eval_")
        self.log_history[-1].update(eval_losses_dict)
        eval_loss = eval_losses_dict[self.args.metric_for_best_model]#.item()
        if self.is_better(eval_loss, self.best_eval_loss):
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            if self.args.save_path is not None:
                self.model.save_model(os.path.join(self.args.save_path))
        else:
            if self.args.early_stopping_patience is not None:
                self.patience_counter += 1
                if self.patience_counter >= self.args.early_stopping_patience:
                    print(f"Early stopping triggered!")
                    return

    def __load_best_model(self):
        """
        Loads the best model from the specified save path if it exists.
        This is useful for resuming training or evaluation with the best
        performing model.
        """
        single_sep_token = self.model.single_sep_token
        dropout_p = self.model.regression_head[0].p
        
        del self.model
        gc.collect()

        self.model = BartWithRegression(
            self.args.save_path,
            single_sep_token=single_sep_token,
            regression_dropout=dropout_p
        ).to(self.device)

    def train(self):
        """
        Trains the model for a specified number of epochs. It handles the
        training loop, validation, and early stopping based on the evaluation
        loss. The training process is logged for monitoring purposes.
        """
        if self.args.greater_is_better:
            self.best_eval_loss = -float('inf')
            self.is_better = lambda x, y: x > y
        else:
            self.best_eval_loss = float('inf')
            self.is_better = lambda x, y: x < y

        self.epochs_bar = tqdm(range(self.args.num_epochs), desc="Training Epochs", position=0)
        # self.epochs_bar.set_postfix_str(
        #     "Epoch {}.{}/{} | lr_body: {:.4e} | lr_head: {:.4e}".format(
        #         self.epochs_bar.n, 0,
        #         self.optimizer.param_groups[0]['lr'],
        #         self.optimizer.param_groups[1]['lr']
        #     )
        # )
        log_df = None
        display_handle = display(log_df, display_id=True)
        for epoch in self.epochs_bar:
            self.__epoch_step()

            if self.eval_dataloader is not None:
                self.__validate_step()

            self.log_history[-1].update({'epoch': epoch + 1})
            if isinstance(self.args.criterion, UncertaintyLoss):
                self.log_history[-1].update({
                    'reg_loss_weight': self.args.criterion.log_var_polarity.item(),
                    'gen_loss_weight': self.args.criterion.log_var_explanation.item()
                })

            if log_df is None:
                log_df = DataFrame([self.log_history[-1]])
            else:
                log_df.loc[len(log_df)] = self.log_history[-1]
            display_handle.update(log_df)