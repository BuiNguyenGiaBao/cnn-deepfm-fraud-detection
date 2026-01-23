import torch
import torch.nn as nn
from cnn_for_extract_feature import TabularCNNNetwork
from deepfm_for_relationship import DeepFM

class HybridCNNDeepFM(nn.Module):
    def __init__(self,tabular_dim: int, 
                #cnn        
                embed_dim = 64,
                conv_channels = 128,
                kernel_size = 3,
                bilinear_rank = 64,
                bilinear_out_dim= 256,
                seq_length= 10,
                cnn_dropout = 0.3,
                # DeepFM params
                num_classes: int = 2,
                deepfm_embed_dim: int = 16,
                deepfm_hidden=None,
                deepfm_dropout: float = 0.2,
                # Training mode
                freeze_cnn: bool = False,):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_cnn = freeze_cnn

        self.cnn = TabularCNNNetwork(tabular_dim=tabular_dim,embed_dim=embed_dim,conv_channels=conv_channels,
            kernel_size=kernel_size,bilinear_rank=bilinear_rank,bilinear_out_dim=bilinear_out_dim,num_classes=num_classes, seq_length=seq_length,dropout=cnn_dropout)

        deepfm_hidden = deepfm_hidden or [256, 128, 64]
        self.deepfm = DeepFM(num_classes=num_classes,categorical_cardinalities=None,num_numerical=0,embed_dim=deepfm_embed_dim,
            deep_hidden=deepfm_hidden,dropout=deepfm_dropout,dense_in_dim=bilinear_out_dim,use_bias=True,)

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn.get_embedding(x, detach=self.freeze_cnn)
        logits = self.deepfm(dense_x=cnn_features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn.get_embedding(x, detach=True)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn.get_attention_weights(x)            
    
    