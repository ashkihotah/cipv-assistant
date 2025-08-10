import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BertForSequenceClassification, 
    BertConfig,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, List, Dict, Union, Tuple

class BertForRegression(BertForSequenceClassification):
    """
    A BERT regression model that leverages the pretrained classifier head.

    Instead of replacing the original classifier, this model keeps it and adds
    a new linear layer on top (`self.regression_head`). This allows the model
    to use the features learned by the original classification head for the new
    regression task.

    The final output is passed through a Tanh function in the forward pass to
    squash the polarity prediction to the [-1, 1] range.

    This model is fully compatible with the Hugging Face Trainer API.
    """

    def __init__(self, config: BertConfig, cls_token_id: int = None):
        super().__init__(config)
        if cls_token_id is None:
            raise ValueError("cls_token_id must be provided to identify the CLS token in input_ids.")
        self.cls_token_id = cls_token_id

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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """
        Forward pass for the regression model.

        Args:
            labels (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the Mean Squared Error regression loss.

        Returns:
            `transformers.modeling_outputs.SequenceClassifierOutput`: An object
            containing the loss (if labels are provided), logits (the final
            polarity predictions), hidden states, and attentions.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Step 1: Get the standard BERT model outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,  # output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states  # tuple: (layer0, layer1, ..., last_layer)
        last_hidden_state = hidden_states[-1]  # shape: (batch_size, seq_len, hidden_size)
        cls_mask = (input_ids == self.cls_token_id)  # shape: (batch_size, seq_len)

        # Step 2: Pass through the original dropout and classifier layers
        # This leverages the pretrained classifier as a feature extractor.
        output = self.dropout(last_hidden_state[cls_mask])
        output = self.classifier(output)
        logits = torch.tanh(output).squeeze()

        loss = None
        if labels is not None:
            # Flatten labels and create a mask for non-padded labels
            flat_labels = labels.view(-1)
            label_mask = flat_labels != -100
            
            # Filter both logits and labels to only include non-padded items
            # The number of True values in cls_mask should equal the number of non-padded labels
            filtered_logits = logits
            filtered_labels = flat_labels[label_mask]

            # Ensure the number of predictions matches the number of valid labels
            # if filtered_logits.shape[0] != filtered_labels.shape[0]:
            #     raise RuntimeError(
            #         f"Mismatch between number of CLS tokens ({filtered_logits.shape[0]}) "
            #         f"and non-padded labels ({filtered_labels.shape[0]}). "
            #         "Check data collator and input data."
            #     )

            loss = F.mse_loss(filtered_logits, filtered_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class DataCollatorForPolarity(DataCollatorWithPadding):
    """
    Data collator for polarity regression tasks.

    This collator extends DataCollatorWithPadding to handle a 'polarity' field
    in the input features. It extracts the polarity values, converts them to a
    torch.FloatTensor, and adds them to the batch dictionary under the key 'labels'.

    This makes it compatible with models that expect a `labels` argument for
    loss calculation, such as the `BertForRegression` model.
    """
    def __init__(self, tokenizer, padding=True, max_length=None, pad_to_multiple_of=None):
        super().__init__(tokenizer, padding=padding, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of)

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collates a batch of features.

        Args:
            features (List[Dict[str, Union[List[int], torch.Tensor]]]): A list of
                feature dictionaries. Each dictionary must contain tokenized inputs
                (e.g., 'input_ids') and a 'polarity' key with a float value.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the padded batch of
                inputs and a 'labels' tensor.
        """
        # Separate the polarities from the features that need padding
        label_tensors = [feature.pop("labels") for feature in features]
        
        # Use the parent class to handle padding of 'input_ids', 'attention_mask', etc.
        batch = super().__call__(features)
        
        padded_labels = pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=-100  # Standard ignore_index for loss calculation
        )
            
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.float)
        
        return batch

