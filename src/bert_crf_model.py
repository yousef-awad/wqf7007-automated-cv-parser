# import dependencies
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel

# define model class
class BertCRFForNER(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.1):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # contextual embeddings for every token -> (batch_size, sequence_length, hidden_size)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        
        outputs = self.bert(**model_inputs)

        sequence_output = self.dropout(outputs.last_hidden_state)           # get token embeddings
        emissions = self.classifier(sequence_output)                        # raw label scores -> (batch_size, sequence_length, num_labels)

        loss = None

        if labels is not None:
            # Use attention_mask as CRF mask so the first timestep ([CLS]) is
            # always "on", satisfying the torchcrf requirement.
            if attention_mask is not None:
                mask = attention_mask.bool()
            else:
                mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)

            # CRF cannot accept -100 labels, so replace ignore labels with 0 (O)
            labels_for_crf = labels.clone()
            labels_for_crf[labels_for_crf == -100] = 0

            # CRF loss -> neg log-likelihood
            loss = -self.crf(
                emissions,
                labels_for_crf,
                mask=mask,
                reduction="mean"
            )

        return {"loss": loss, "logits": emissions}          # compatible for hf Trainer
    
    def decode(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """Use during inference."""
        
        self.eval()

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        
        outputs = self.bert(**model_inputs)

        sequence_output = self.dropout(outputs.last_hidden_state) 
        emissions = self.classifier(sequence_output)

        # Always use attention_mask for CRF decode so [CLS] (position 0)
        # is included — torchcrf requires the first timestep to be unmasked.
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        decoded = self.crf.decode(emissions=emissions, mask=mask)       # best valid sequence across tokens

        return decoded