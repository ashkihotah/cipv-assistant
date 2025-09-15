import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, List, Dict, Union, Tuple, Any
from dataclasses import dataclass

from transformers import (
    BertConfig,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings, 
    BertModel, 
    BertForSequenceClassification,
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_attention_mask_for_sdpa
)

@dataclass
class DataCollatorWithUserTypePadding(DataCollatorWithPadding):
    """
    Custom data collator that extends DataCollatorWithPadding to handle user_type_ids.
    """
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Handle user_type_ids separately
        user_type_ids = [feature.pop("user_type_ids") for feature in features if "user_type_ids" in feature]
        
        # Use the parent class to handle standard padding
        batch = super().__call__(features)
        
        # Manually pad user_type_ids to match the padded input_ids length
        if user_type_ids:
            max_length = batch["input_ids"].shape[1]  # Get the padded length
            padded_user_type_ids = []
            
            for ids in user_type_ids:
                # Convert to tensor if it's not already
                if not isinstance(ids, torch.Tensor):
                    ids = torch.tensor(ids, dtype=torch.long)
                
                # Pad with 0s (assuming 0 is a valid padding token for user types)
                if len(ids) < max_length:
                    padding_length = max_length - len(ids)
                    ids = torch.cat([ids, torch.zeros(padding_length, dtype=torch.long)])
                
                padded_user_type_ids.append(ids)
            
            batch["user_type_ids"] = torch.stack(padded_user_type_ids)
        
        return batch

class BertEmbeddingsWithUserType(BertEmbeddings):
    def __init__(self, config):
        if not hasattr(config, 'user_type_vocab_size'):
            raise ValueError("config must have 'user_type_vocab_size' attribute.")

        super().__init__(config)

        self.user_type_embeddings = nn.Embedding(
            config.user_type_vocab_size, config.hidden_size
        )

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, 
            inputs_embeds=None, past_key_values_length=0, user_type_ids=None
        ):

        # ==== v COPIED EXACTLY FROM THE SUPER CLASS v ====

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # ==== ^ COPIED EXACTLY FROM THE SUPER CLASS ^ ====      

        # First, create the base embeddings just like the parent class does.
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
            
        # Get the new user_type embeddings
        # We need to make sure user_type_ids is provided.
        if user_type_ids is None:
            raise ValueError("user_type_ids must be provided to BertEmbeddingsWithUserType")
        user_type_embeddings = self.user_type_embeddings(user_type_ids)
        embeddings += user_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModelWithUserType(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # Custom behavior: Replace the standard embeddings with our custom version.
        self.embeddings = BertEmbeddingsWithUserType(config)

    # Method exactly copied from the superclass, i only added user_type_ids
    # instructions with comment "custom behaviour" are the only differentiators
    # all other instructions are merely copied by the super().forward() method code
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        user_type_ids: Optional[torch.Tensor] = None, # custom behaviour
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            user_type_ids=user_type_ids, # custom behaviour
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class BertWithUserTypeForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # Custom behavior: Replace the standard `self.bert` with our custom model.
        self.bert = BertModelWithUserType(config)

    def _init_weights(self, module):
        # First, let the standard initializer do its work for all other layers.
        super()._init_weights(module)
        
        # Now, specifically handle our custom user_type_embeddings.
        if isinstance(module, BertEmbeddingsWithUserType):
            # Option 1 (Best): Copy token_type_embeddings
            # print("Initializing user_type_embeddings by copying token_type_embeddings.")
            # with torch.no_grad():
            #     # Get the pre-trained token_type_embeddings from the model
            #     token_embeddings = self.bert.embeddings.token_type_embeddings.weight
                
            #     # Check if dimensions are compatible before copying
            #     if module.user_type_embeddings.weight.shape == token_embeddings.shape:
            #         module.user_type_embeddings.weight.copy_(token_embeddings)
            #     else:
            #         print("Warning: Shape mismatch, falling back to random initialization.")
            #         std = self.config.initializer_range
            #         module.user_type_embeddings.weight.data.normal_(mean=0.0, std=std)

            # --- OR ---

            # # Option 2 (Excellent Alternative): Small random initialization
            print(f"Initializing user_type_embeddings with Normal(0, {self.config.initializer_range}).")
            std = self.config.initializer_range
            module.user_type_embeddings.weight.data.normal_(mean=0.0, std=std)
    
    # Method exactly copied from the superclass, i only added user_type_ids
    # instructions with comment "custom behaviour" are the only differentiators
    # all other instructions are merely copied by the super().forward() method code
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        user_type_ids: Optional[torch.Tensor] = None # custom behavior
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            user_type_ids=user_type_ids # custom behavior
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )