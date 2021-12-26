import torch.nn as nn

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
