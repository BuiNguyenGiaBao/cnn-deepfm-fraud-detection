import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # Transpose: (batch, length, channels)
        x_t = x.transpose(1, 2)
        
        # Attention scores: (batch, length, 1)
        attn_scores = self.attention(x_t)
        
        # Softmax: (batch, length, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted sum: (batch, channels)
        pooled = torch.sum(x * attn_weights.transpose(1, 2), dim=2)
        
        return pooled, attn_weights


class LowRankBilinear(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, rank):
        super(LowRankBilinear, self).__init__()
        self.rank = rank
        
        # Low-rank factorization
        self.U1 = nn.Linear(in1_features, rank, bias=False)
        self.U2 = nn.Linear(in2_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features)
    
    def forward(self, x1, x2):
        # Project to low-rank space
        u1 = self.U1(x1)  # (batch, rank)
        u2 = self.U2(x2)  # (batch, rank)
        
        # Element-wise multiplication (Hadamard product)
        interaction = u1 * u2  # (batch, rank)
        
        # Project to output space
        output = self.V(interaction)  # (batch, out_features)
        
        return output


class TabularCNNNetwork(nn.Module):
    def __init__(self,tabular_dim,embed_dim,conv_channels,kernel_size,bilinear_rank,bilinear_out_dim,num_classes,seq_length=10,dropout=0.3):
        super(TabularCNNNetwork, self).__init__()
        self.tabular_dim = tabular_dim
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.conv_channels = conv_channels
        self.bilinear_rank = bilinear_rank
        
        # 1. Tabular embedding layer
        self.tabular_embed = nn.Sequential(
            nn.Linear(tabular_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Reshape to sequence
        self.feature_projection = nn.Linear(embed_dim, embed_dim * seq_length)
        
        # 2. CNN layers
        self.conv1 = nn.Conv1d(embed_dim, conv_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        
        # 3. Attention pooling
        self.attention_pool = AttentionPooling(conv_channels)
        
        # 4. Low-rank bilinear pooling
        self.low_rank_bilinear = LowRankBilinear(
            in1_features=conv_channels,
            in2_features=conv_channels,
            out_features=bilinear_out_dim,
            rank=bilinear_rank
        )
        
        # 5. Output layer
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(bilinear_out_dim, num_classes)
        

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Tabular embedding
        embedded = self.tabular_embed(x)  # (batch, embed_dim)
        
        # Project and reshape to sequence
        projected = self.feature_projection(embedded)  # (batch, embed_dim * seq_length)
        seq = projected.view(batch_size, self.seq_length, -1)  # (batch, seq_length, embed_dim)
        seq = seq.transpose(1, 2)  # (batch, embed_dim, seq_length)
        
        # 2. CNN layers
        conv1_out = F.relu(self.bn1(self.conv1(seq)))  # (batch, conv_channels, seq_length)
        conv2_out = F.relu(self.bn2(self.conv2(conv1_out)))  # (batch, conv_channels, seq_length)
        
        # 3. Attention pooling
        pooled, attn_weights = self.attention_pool(conv2_out)  # (batch, conv_channels)
        
        # 4. Low-rank bilinear (self-interaction)
        bilinear_out = self.low_rank_bilinear(pooled, pooled)  # (batch, bilinear_out_dim)
        bilinear_out = F.relu(bilinear_out)
        
        # 5. Output
        bilinear_out = self.dropout(bilinear_out)
        logits = self.output(bilinear_out)  # (batch, num_classes)
        
        return logits, attn_weights
    
    def predict(self, x):

        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def get_attention_weights(self, x):
        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward(x)
        return attn_weights.squeeze(-1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Tabular embedding
        embedded = self.tabular_embed(x)

        # Sequence projection
        projected = self.feature_projection(embedded)
        seq = projected.view(batch_size, self.seq_length, -1)
        seq = seq.transpose(1, 2)

        # CNN
        conv1_out = F.relu(self.bn1(self.conv1(seq)))
        conv2_out = F.relu(self.bn2(self.conv2(conv1_out)))

        # Attention pooling
        pooled, _ = self.attention_pool(conv2_out)

        # Bilinear
        embedding = self.low_rank_bilinear(pooled, pooled)
        embedding = F.relu(embedding)

        return embedding

    def get_embedding(self, x: torch.Tensor, detach: bool = True) -> torch.Tensor:
        if detach:
            with torch.no_grad():
                return self.compute_embedding(x)
        return self.compute_embedding(x)