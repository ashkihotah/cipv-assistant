import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
from transformers.modeling_outputs import ModelOutput, Seq2SeqLMOutput

@dataclass
class Seq2SeqLMWithRegressionOutput(Seq2SeqLMOutput):
    """
    Extends `Seq2SeqLMOutput` to include outputs for a regression task.

    Args:
        generation_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss from the base BART model.
        regression_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Regression loss (e.g., MSE) for the polarity prediction task.
        regression_scores (`torch.FloatTensor` of shape `(batch_size, num_sep_tokens)`):
            Prediction scores of the regression head.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            The total loss, which is a combination of generation and regression losses.
            This overrides the `loss` from the parent class to ensure it's documented
            as the combined loss.
    """
    # Note: logits, past_key_values, all hidden_states and attentions are inherited
    # from Seq2SeqLMOutput and do not need to be re-declared.
    
    regression_scores: torch.FloatTensor = None
    generation_loss: Optional[torch.FloatTensor] = None
    regression_loss: Optional[torch.FloatTensor] = None
    
    # We can re-declare `loss` to control its order in the __init__ signature
    # and to provide a more specific docstring, but it's not strictly necessary.
    loss: Optional[torch.FloatTensor] = None

class BartForConditionalGenerationAndRegression(BartForConditionalGeneration):
    """
    A BART model that performs both conditional text generation and regression on
    special separator tokens within the encoder's hidden states.
    """
    def __init__(self, config: BartConfig):
        """
        Initializes the model. The configuration object should be extended with
        custom parameters before being passed to this constructor.

        Args:
            config (BartConfig): The model configuration. Expected to have
                additional attributes:
                - `regression_dropout` (float): Dropout probability for the regression head.
                - `single_sep_token` (bool): Whether to use one or two special separator tokens.
        """
        super().__init__(config)

        # Ensure custom config attributes exist
        if not hasattr(config, 'regression_dropout'):
            raise ValueError("config must have 'regression_dropout' attribute.")
        if not hasattr(config, 'single_sep_token'):
            raise ValueError("config must have 'single_sep_token' attribute.")
        if not hasattr(config, 'sep_token_id'):
            raise ValueError("config must have 'sep_token_id' attribute.")
        if not config.single_sep_token and not hasattr(config, 'sep_token_id2'):
            raise ValueError("config must have 'sep_token_id2' attribute for dual-separator mode.")
            
        self.single_sep_token = config.single_sep_token

        # Define the custom regression head
        self.regression_head = nn.Sequential(
            nn.Dropout(config.regression_dropout),
            nn.Linear(config.hidden_size, 1),
            nn.Tanh()
        )
        
        # Initialize weights for the new head
        self._init_weights(self.regression_head)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        regression_labels: Optional[torch.FloatTensor] = None # Custom Behaviour
    ) -> Union[Tuple, Seq2SeqWithRegressionOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Standard BART forward pass to get encoder-decoder outputs
        # We need the encoder's last hidden state for the regression task
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, # Custom behavior: Ensure hidden states are returned
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # ==== Regression Task ====
        encoder_last_hidden_state = outputs.encoder_last_hidden_state

        # Create a mask to find the locations of the separator tokens
        sep_token_mask = (input_ids == self.config.sep_token_id)
        if not self.single_sep_token:
            sep_token_mask |= (input_ids == self.config.sep_token_id2)

        # ==== if polarities are given as a 2-D tensor with shape (batch_size, n_polarities) ====
        # # Extract the hidden states corresponding to the separator tokens
        # sep_hidden_states = encoder_last_hidden_state[sep_token_mask]
        # # Pass through the regression head
        # regression_scores = self.regression_head(sep_hidden_states).squeeze(-1)

        # ==== if polarities are given as a flattened 1-D tensor with length batch_size x n_polarities ====
        flattened_hidden_states = encoder_last_hidden_state.view(-1, encoder_last_hidden_state.size(-1))
        flattened_sep_mask = sep_token_mask.view(-1)
        regression_scores = self.regression_head(flattened_hidden_states[flattened_sep_mask]).squeeze(-1)

        # ==== Generation Task ====
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        # ==== Loss Calculation ====
        total_loss = None
        generation_loss = None
        regression_loss = None

        # Calculate generation loss
        if labels is not None:
            generation_loss = F.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        # Calculate regression loss
        if regression_labels is not None:
            # The number of predicted scores must match the number of labels
            if regression_scores.shape[0] != regression_labels.shape[0]:
                raise ValueError(
                    f"The number of predicted regression scores ({regression_scores.shape[0]}) "
                    f"does not match the number of regression labels ({regression_labels.shape[0]}). "
                    "Ensure that the number of SEP tokens in the input matches the number of labels provided."
                )
            regression_loss = F.mse_loss(regression_scores, regression_labels)

        # Combine losses if both are computed
        if generation_loss is not None and regression_loss is not None:
            # Here, we do a simple sum. A more complex weighting strategy (like UncertaintyLoss)
            # should be handled by a custom Trainer class.
            total_loss = generation_loss + regression_loss
        elif generation_loss is not None:
            total_loss = generation_loss
        elif regression_loss is not None:
            total_loss = regression_loss

        if not return_dict:
            output = (lm_logits, regression_scores) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqLMWithRegressionOutput(
            loss=total_loss,
            generation_loss=generation_loss,
            regression_loss=regression_loss,
            logits=lm_logits,
            regression_scores=regression_scores,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

def setup_model_and_tokenizer(
    model_name_or_path: str,
    regression_dropout: float,
    single_sep_token: bool,
    **kwargs
) -> (BartForConditionalGenerationAndRegression, BartTokenizer):
    """
    Helper function to correctly initialize the tokenizer with special tokens
    and configure the model.
    """
    # Define special tokens
    SEP_TOKEN = "[SEP]"
    USR0_TOKEN = "[USR0]"
    USR1_TOKEN = "[USR1]"
    
    # 1. Initialize tokenizer and add special tokens
    tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    if single_sep_token:
        special_tokens_dict = {"additional_special_tokens": [SEP_TOKEN]}
    else:
        special_tokens_dict = {"additional_special_tokens": [USR0_TOKEN, USR1_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    # 2. Get config and add custom attributes
    config = BartConfig.from_pretrained(model_name_or_path, **kwargs)
    config.regression_dropout = regression_dropout
    config.single_sep_token = single_sep_token
    if single_sep_token:
        config.sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    else:
        config.sep_token_id = tokenizer.convert_tokens_to_ids(USR0_TOKEN)
        config.sep_token_id2 = tokenizer.convert_tokens_to_ids(USR1_TOKEN)

    # 3. Initialize the model with the extended config
    model = BartForConditionalGenerationAndRegression.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True # Important for adding new tokens
    )

    # 4. Resize token embeddings to match the new tokenizer size
    model.resize_token_embeddings(len(tokenizer))
    
    # (Optional) Initialize new token embeddings if desired
    # For example, to initialize [SEP] as the mean of <s> and </s>
    # with torch.no_grad():
    #     emb = model.get_input_embeddings()
    #     sep_id = config.sep_token_id
    #     bos_id = tokenizer.bos_token_id
    #     eos_id = tokenizer.eos_token_id
    #     mean_emb = (emb.weight[bos_id] + emb.weight[eos_id]) / 2
    #     emb.weight[sep_id] = mean_emb
        
    return model, tokenizer

@dataclass # for giving padded 1-D flattened regression labels tensor
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
        # Extract regression labels from features and create a 1-D tensor
        # containing all regression labels concatenated
        regression_labels = [
            label for feature in features
            for label in feature.pop('regression_labels')
        ]

        # Let the parent class handle the standard seq2seq padding and
        # creation of decoder_input_ids.
        # This will correctly pad input_ids, attention_mask, and labels.
        batch = super().__call__(features, return_tensors)

        # Convert regression labels to a tensor
        batch['regression_labels'] = torch.tensor(regression_labels, dtype=torch.float32)
        return batch

# @dataclass # for giving padded 2-D regression labels tensor
# class DataCollatorForBARTMultiTask(DataCollatorForSeq2Seq):
#     """
#     Extends `DataCollatorForSeq2Seq` to also handle custom `regression_labels`.
#     This collator pads `input_ids`, `attention_mask`, and `labels` as usual for
#     seq2seq tasks, and additionally pads `regression_labels` to the maximum
#     length in the batch using -100 as the padding value.
#     """
#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]], return_tensors=None) -> Dict[str, torch.Tensor]:
#         # Pop custom regression labels before the parent collator processes the features.
#         regression_labels_list = [
#             torch.tensor(feature.pop("regression_labels"), dtype=torch.float)
#             for feature in features if "regression_labels" in feature
#         ]
        
#         # Let the parent class handle standard seq2seq padding for inputs and generation labels.
#         batch = super().__call__(features, return_tensors)
        
#         # If regression labels were present, pad them and add to the final batch.
#         if regression_labels_list:
#             padded_regression_labels = pad_sequence(
#                 regression_labels_list,
#                 batch_first=True,
#                 padding_value=-100  # Standard ignore_index for loss calculation
#             )
#             batch["regression_labels"] = padded_regression_labels
            
#         return batch

# bart with a body like my custom bert body for single target