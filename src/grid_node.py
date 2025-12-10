
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerGridNode(nn.Module):
    """
    A computational abstraction of a neuron[cite: 38].
    Implements Dendritic Integration, Soma Computation, and Axonal Broadcast.
    """
    def __init__(self, d_model, nhead, dim_feedforward, threshold=0.0):
        super(TransformerGridNode, self).__init__()
        
        # Dendritic Input: Aggregates signals [cite: 39]
        self.input_projection = nn.Linear(d_model, d_model)
        
        # Soma Computation: Layer Norm + Attention + MLP [cite: 50]
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.threshold = threshold

    def forward(self, neighbor_signals):
        # 1. Integration (Many-to-One) [cite: 43]
        if not neighbor_signals:
            return torch.zeros(1)
        stacked_inputs = torch.stack(neighbor_signals, dim=0) 
        integrated_input = torch.sum(stacked_inputs, dim=0) # [cite: 17]
        
        # 2. Processing (Soma) [cite: 40]
        x = self.norm1(integrated_input)
        attn_output, _ = self.self_attn(x, x, x) 
        x = integrated_input + attn_output
        x_norm = self.norm2(x)
        processed_signal = x + self.mlp(x_norm)

        # 3. Thresholding (Spike Generation) [cite: 31]
        signal_magnitude = torch.norm(processed_signal, p=2, dim=-1, keepdim=True)
        spike_mask = (signal_magnitude > self.threshold).float()
        
        # 4. Broadcast (One-to-Many) [cite: 44]
        return processed_signal * spike_mask
