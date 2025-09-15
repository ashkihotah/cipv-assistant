import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict, Any

from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    BartTokenizer,
    BartModel,
    BartEncoder,
    DataCollatorForSeq2Seq
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding, BartAttention, _expand_mask
from transformers.utils import logging

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

# Use the same custom output dataclass from your bart.py
# to maintain consistency in handling multitask outputs.
@dataclass
class Seq2SeqLMWithRegressionOutput(Seq2SeqLMOutput):
    regression_scores: torch.FloatTensor = None
    generation_loss: Optional[torch.FloatTensor] = None
    regression_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None

class BartEmbeddingsWithUserType(nn.Module):
    """
    Constructs the embeddings for BART's encoder, adding support for
    token_type_ids and user_type_ids, mirroring the logic from
    BertEmbeddingsWithUserType.
    """
    def __init__(self, config: BartConfig):
        super().__init__()
        
        # Ensure custom config attributes exist
        if not hasattr(config, 'user_type_vocab_size'):
            raise ValueError("config must have 'user_type_vocab_size' attribute.")
        # Standard BART doesn't have token_type_ids, so we check for its vocab size too
        if not hasattr(config, 'type_vocab_size'):
            config.type_vocab_size = 2 # Default value like in BERT

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.position_embeddings = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.user_type_embeddings = nn.Embedding(config.user_type_vocab_size, config.hidden_size)

        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        user_type_ids: torch.LongTensor,
        past_key_values_length: int = 0,
    ):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # 1. Get word embeddings
        inputs_embeds = self.word_embeddings(input_ids)

        # 2. Get position embeddings
        position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        position_embeddings = self.position_embeddings(input_ids, past_key_values_length)

        # 3. Get token type embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # 4. Get user type embeddings
        user_type_embeddings = self.user_type_embeddings(user_type_ids)

        # 5. Sum all embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + user_type_embeddings
        
        # 6. Apply LayerNorm and Dropout
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BartEncoderWithUserType(BartEncoder):
    """
    A BART Encoder that uses BartEmbeddingsWithUserType to incorporate
    token_type_ids and user_type_ids.
    The original standard embedding layers (embed_tokens, embed_positions) 
    are removed/overridden.
    """
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        
        # 1. Initialize parent. We pass embed_tokens=None here if possible, 
        # but since BartModelWithUserType passes 'self.shared', we accept it 
        # but immediately discard the inherited variables.
        # Note: We must still call super().__init__ to set up encoder layers, 
        # layer norm, and other BartEncoder infrastructure.
        super().__init__(config, embed_tokens=embed_tokens)
        
        # 2. Replace standard embeddings with our custom version
        self.embeddings = BartEmbeddingsWithUserType(config)
        
        # 3. Explicitly remove/nullify inherited standard BART embedding components.
        # In this custom structure, the role of shared/positional embeddings 
        # is entirely delegated to self.embeddings.
        if hasattr(self, 'embed_tokens'):
            # Delete the reference entirely instead of setting to None
            del self.embed_tokens
        if hasattr(self, 'embed_positions'):
            del self.embed_positions
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None, # Custom
        user_type_ids: Optional[torch.LongTensor] = None,  # Custom
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        
        # Standard BART forward setup
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("`input_ids` must be provided to BartEncoderWithUserType.")
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be provided to BartEncoderWithUserType.")
        if user_type_ids is None:
            raise ValueError("`user_type_ids` must be provided to BartEncoderWithUserType.")

        # === Custom Behavior: Use our custom embedding layer ===
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            user_type_ids=user_type_ids
        )
        # =======================================================
        
        # The rest of the forward pass is standard for a Transformer encoder
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        attention_mask = _expand_mask(attention_mask, embedding_output.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        hidden_states = embedding_output
        
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
            
        return BaseModelOutput(
            last_hidden_state=hidden_states, 
            hidden_states=encoder_states, 
            attentions=all_attentions
        )

class BartModelWithUserType(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        # Replace the standard encoder with our custom version
        self.encoder = BartEncoderWithUserType(config, self.shared)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None, # Custom
        user_type_ids: Optional[torch.LongTensor] = None,  # Custom
        # ... other standard BartModel arguments
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # We only need to override the call to the encoder, passing the new arguments.
        # The decoder's logic remains unchanged.
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids, # Pass to custom encoder
                user_type_ids=user_type_ids,   # Pass to custom encoder
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # The rest of the method is identical to the original BartModel.forward
        return super().forward(
            input_ids=None, # Already processed by encoder
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=None,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class BartForConditionalGenerationAndRegressionWithUserType(BartForConditionalGeneration):
    """
    A BART model for multi-task learning: conditional text generation and regression.
    This model uses a custom encoder that accepts `token_type_ids` and `user_type_ids`
    to better distinguish between context and target sentences, similar to the
    provided custom BERT implementation.
    """
    def __init__(self, config: BartConfig):
        super().__init__(config)

        # Replace the standard BartModel with our custom version
        self.model = BartModelWithUserType(config)

        # Ensure custom config attributes exist for the regression head
        if not hasattr(config, 'regression_dropout'):
            config.regression_dropout = 0.1 # Provide a default
        if not hasattr(config, 'sep_token_id'):
            raise ValueError("config must have 'sep_token_id' attribute.")
            
        # Define the custom regression head
        self.regression_head = nn.Sequential(
            nn.Dropout(config.regression_dropout),
            nn.Linear(config.hidden_size, 1),
            nn.Tanh()
        )
        
        # Initialize weights for the new head and custom embeddings
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None, # Custom
        user_type_ids: Optional[torch.LongTensor] = None,  # Custom
        regression_labels: Optional[torch.FloatTensor] = None, # Custom
        # ... other standard arguments
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMWithRegressionOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Standard forward pass using our custom BartModel
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, # Pass custom args
            user_type_ids=user_type_ids,   # Pass custom args
            decoder_input_ids=decoder_input_ids,
            labels=labels, # Pass labels to model for decoder_input_ids shifting
            output_hidden_states=True, # Ensure we get encoder hidden states
            return_dict=return_dict,
            **kwargs,
        )
        
        # ==== Regression Task ====
        # This logic is similar to your BertForMultipleRegressions
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        sep_mask = (input_ids == self.config.sep_token_id)
        
        # Extract hidden states corresponding to the SEP tokens
        sep_hidden_states = encoder_last_hidden_state[sep_mask]
        
        # Pass through the regression head
        regression_scores = self.regression_head(sep_hidden_states).squeeze(-1)

        # ==== Generation Task ====
        lm_logits = F.linear(outputs.last_hidden_state, self.model.shared.weight, bias=self.final_logits_bias)

        # ==== Loss Calculation ====
        total_loss = None
        generation_loss = None
        regression_loss = None

        if labels is not None:
            generation_loss = F.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=self.config.pad_token_id)
        
        if regression_labels is not None:
            if regression_scores.shape[0] != regression_labels.shape[0]:
                raise ValueError(
                    f"Mismatch between number of SEP tokens found ({regression_scores.shape[0]}) "
                    f"and regression labels provided ({regression_labels.shape[0]})."
                )
            regression_loss = F.mse_loss(regression_scores, regression_labels)

        # Combine losses
        if generation_loss is not None and regression_loss is not None:
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
    user_type_vocab_size: int,
    regression_dropout: float = 0.1,
    **kwargs
) -> (BartForConditionalGenerationAndRegressionWithUserType, BartTokenizer):
    """
    Helper function to correctly initialize the tokenizer with a special token
    and configure the custom model.
    """
    SEP_TOKEN = "[SEP]"
    
    # 1. Initialize tokenizer and add special token
    tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": [SEP_TOKEN]})

    # 2. Get config and add custom attributes
    config = BartConfig.from_pretrained(model_name_or_path, **kwargs)
    config.regression_dropout = regression_dropout
    config.user_type_vocab_size = user_type_vocab_size
    config.type_vocab_size = 2  # For segment A vs segment B
    config.sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    # 3. Initialize the model with the extended config
    model = BartForConditionalGenerationAndRegressionWithUserType.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True # Important for adding new tokens and layers
    )

    # 4. Resize token embeddings to match the new tokenizer size
    model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer