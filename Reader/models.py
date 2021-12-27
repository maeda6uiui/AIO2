import torch
import torch.nn as nn
from transformers import BertConfig,BertModel

class PlausibilityCalculator(nn.Module):
    def __init__(self,dim_feature_vector:int=768):
        super().__init__()

        self.main=nn.Sequential(
            nn.Linear(dim_feature_vector,int(dim_feature_vector/2)),
            nn.Mish(),
            nn.Linear(int(dim_feature_vector/2),100),
            nn.Mish(),
            nn.Linear(100,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.main(x)
        return x

class Reader(nn.Module):
    def __init__(self,bert_model_name:str):
        super().__init__()

        self.bert=BertModel.from_pretrained(bert_model_name)

        config=BertConfig.from_pretrained(bert_model_name)

        self.seq_start=nn.Sequential(
            nn.Linear(config.hidden_size*4,config.hidden_size*2),
            nn.Mish(),
            nn.Linear(config.hidden_size*2,config.hidden_size),
            nn.Mish(),
            nn.Linear(config.hidden_size,256),
            nn.Mish(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
        self.seq_end=nn.Sequential(
            nn.Linear(config.hidden_size*4,config.hidden_size*2),
            nn.Mish(),
            nn.Linear(config.hidden_size*2,config.hidden_size),
            nn.Mish(),
            nn.Linear(config.hidden_size,256),
            nn.Mish(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
        self.seq_plausibility=nn.Sequential(
            nn.Linear(config.hidden_size*4,config.hidden_size*2),
            nn.Mish(),
            nn.Linear(config.hidden_size*2,config.hidden_size),
            nn.Mish(),
            nn.Linear(config.hidden_size,256),
            nn.Mish(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(
        self,
        input_ids:torch.LongTensor,
        attention_mask:torch.LongTensor,
        token_type_ids:torch.LongTensor,
        start_positions:torch.LongTensor=None,
        end_positions:torch.LongTensor=None):
        bert_inputs={
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "token_type_ids":token_type_ids,
            "return_dict":True,
            "output_hidden_states":True
        }
        bert_outputs=self.bert(**bert_inputs)

        hidden_states=bert_outputs["hidden_states"][-4:]

        concat_hidden_states=hidden_states[0]   #(N, sequence_length, hidden_size) → (N, sequence_length, hidden_size*4)
        for i in range(1,4):
            concat_hidden_states=torch.cat([concat_hidden_states,hidden_states[i]],dim=2)

        start_logits=self.seq_start(concat_hidden_states)   #(N, sequence_length, 1)
        start_logits=torch.squeeze(start_logits)    #(N, sequence_length)

        end_logits=self.seq_end(concat_hidden_states)
        end_logits=torch.squeeze(end_logits)

        cls_vectors=concat_hidden_states[:,0,:] #(N, hidden_size)
        plausibility_scores=self.seq_plausibility(cls_vectors)  #(M, 1)
        plausibility_scores=torch.squeeze(plausibility_scores)  #(N)

        loss=None
        loss_start=None
        loss_end=None
        loss_plausibility=None
        if start_positions is not None and end_positions is not None:
            criterion_span=nn.CrossEntropyLoss()
            criterion_plausibility=nn.BCELoss()

            batch_size=start_positions.size(0)

            loss_start=0
            loss_end=0
            for i in range(batch_size):
                if start_positions[i]!=0 and end_positions[i]!=0:
                    this_start_logits=torch.unsqueeze(start_logits[i],0)    #(1, sequence_length)
                    this_end_logits=torch.unsqueeze(end_logits[i],0)    #(1, sequence_length)

                    start_position=torch.unsqueeze(start_positions[i],0)    #(1)
                    end_position=torch.unsqueeze(end_positions[i],0)    #(1)

                    loss_start+=criterion_span(this_start_logits,start_position)
                    loss_end+=criterion_span(this_end_logits,end_position)

            plausibility_targets=(start_positions!=0).float()
            loss_plausibility=criterion_plausibility(plausibility_scores,plausibility_targets)

            loss=loss_start+loss_end+loss_plausibility
            loss_start=loss_start.item()
            loss_end=loss_end.item()
            loss_plausibility=loss_plausibility.item()

        ret={
            "start_logits":start_logits,
            "end_logits":end_logits,
            "plausibility_scores":plausibility_scores,

            "loss":loss,
            "loss_start":loss_start,
            "loss_end":loss_end,
            "loss_plausibility":loss_plausibility
        }
        return ret
