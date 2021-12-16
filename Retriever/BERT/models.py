import torch
import torch.nn as nn

class RelevanceScoreCalculator(nn.Module):
    def __init__(self,dim_feature_vector:int=768):
        super().__init__()

        self.main=nn.Sequential(
            nn.Linear(dim_feature_vector*2,dim_feature_vector),
            nn.Mish(),
            nn.Linear(dim_feature_vector,int(dim_feature_vector/2)),
            nn.Mish(),
            nn.Linear(int(dim_feature_vector/2),100),
            nn.Mish(),
            nn.Linear(100,1),
            nn.Sigmoid()
        )

    def forward(self,vec_q:torch.FloatTensor,vec_d:torch.FloatTensor):
        #vec_q: (N, dim_feature_vector)
        #vec_d: (N, dim_feature_vector)

        x=torch.cat([vec_q,vec_d],dim=1)    #(N, dim_feature_vector*2)
        x=self.main(x)  #(N, 1)

        return x
