import torch
import torch.nn as nn
from transformers import AutoConfig,AutoModel

class Reader(nn.Module):
    def __init__(self,model_name:str):
        super().__init__()

        self.bert=AutoModel.from_pretrained(model_name)

        config=AutoConfig.from_pretrained(model_name)

        self.seq_span=nn.Sequential(
            nn.Linear(config.hidden_size*4,config.hidden_size*2),
            nn.Mish(),
            nn.Linear(config.hidden_size*2,config.hidden_size),
            nn.Mish(),
            nn.Dropout(p=0.1),
            nn.Linear(config.hidden_size,2)
        )
        self.seq_plausibility=nn.Sequential(
            nn.Linear(config.hidden_size*4,config.hidden_size*2),
            nn.Mish(),
            nn.Linear(config.hidden_size*2,config.hidden_size),
            nn.Mish(),
            nn.Dropout(p=0.1),
            nn.Linear(config.hidden_size,1)
        )

    def forward(
        self,
        input_ids:torch.LongTensor, #(N, sequence_length)
        attention_mask:torch.LongTensor,    #(N, sequence_length)
        token_type_ids:torch.LongTensor,    #(N, sequence_length)
        start_positions:torch.LongTensor=None,    #(N, max_num_answer_ranges)
        end_positions:torch.LongTensor=None #(N, max_num_answer_ranges)
    ):
        batch_size=input_ids.size(0)
        sequence_length=input_ids.size(1)

        bert_inputs={
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "token_type_ids":token_type_ids,
            "return_dict":True,
            "output_hidden_states":True
        }
        bert_outputs=self.bert(**bert_inputs)

        hidden_states=bert_outputs["hidden_states"][-4:]

        concat_hidden_states=hidden_states[0]
        for i in range(1,4):
            concat_hidden_states=torch.cat([concat_hidden_states,hidden_states[i]],dim=2)

        span_logits=self.seq_span(concat_hidden_states) #(N, sequence_length, 2)
        start_logits,end_logits=torch.split(span_logits,1,dim=2)    #(N, sequence_length, 1)

        start_logits=start_logits.view(batch_size,sequence_length)    #(N, sequence_length)
        end_logits=end_logits.view(batch_size,sequence_length)    #(N, sequence_length)

        cls_vectors=concat_hidden_states[:,0,:] #(N, hidden_size*4)
        plausibility_scores=self.seq_plausibility(cls_vectors)   #(N, 1)
        plausibility_scores=plausibility_scores.view(-1)  #(N)

        loss=None
        loss_span=None
        loss_plausibility=None
        if start_positions is not None and end_positions is not None:
            criterion_span=nn.CrossEntropyLoss()
            criterion_plausibility=nn.BCEWithLogitsLoss()

            #Span Lossの計算
            max_num_answer_ranges=start_positions.size(1)

            loss_span=0
            for i in range(batch_size):
                this_start_logits=torch.unsqueeze(start_logits[i],0)    #(1, sequence_length)
                this_end_logits=torch.unsqueeze(end_logits[i],0)    #(1, sequence_length)

                for j in range(max_num_answer_ranges):
                    if start_positions[i,j]==0 and end_positions[i,j]==0:
                        continue

                    start_position=torch.unsqueeze(start_positions[i,j],0)    #(1)
                    end_position=torch.unsqueeze(end_positions[i,j],0)    #(1)

                    loss_start=criterion_span(this_start_logits,start_position)
                    loss_end=criterion_span(this_end_logits,end_position)
                    loss_span+=loss_start+loss_end

            loss_span/=batch_size

            #Plausibility Lossの計算
            plausibility_targets=(start_positions[:,0]!=0).float().to(plausibility_scores.device)   #(N)
            loss_plausibility=criterion_plausibility(plausibility_scores,plausibility_targets)
            loss_plausibility*=10

            loss=loss_span+loss_plausibility
            loss_span=loss_span.item() if loss_span!=0 else 0
            loss_plausibility=loss_plausibility.item()

        ret={
            "start_logits":start_logits,
            "end_logits":end_logits,
            "plausibility_scores":plausibility_scores,

            "loss":loss,
            "loss_span":loss_span,
            "loss_plausibility":loss_plausibility
        }
        return ret
