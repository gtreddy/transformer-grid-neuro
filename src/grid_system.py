import torch
import torch.nn as nn
from .grid_node import TransformerGridNode

class SpecialistNode(TransformerGridNode):
    """
    Extends the node to support specialized roles: Memory, Expert, Router.
    """
    def __init__(self, d_model, nhead, role='router'):
        # Experts have higher capacity (Mixture-of-Experts style) [cite: 77]
        dim_ff = d_model * 8 if role == 'expert' else d_model * 2
        super().__init__(d_model, nhead, dim_feedforward=dim_ff, threshold=0.0)
        self.role = role
        if self.role == 'memory':
            self.register_buffer('hidden_memory', torch.zeros(1, 1, d_model))

    def forward(self, neighbor_signals):
        # Reuse base logic with added memory persistence
        if not neighbor_signals:
            integrated = torch.zeros(1, 1, self.input_projection.in_features)
        else:
            integrated = torch.sum(torch.stack(neighbor_signals, dim=0), dim=0)

        x = self.norm1(integrated)
        attn, _ = self.self_attn(x, x, x)
        x = integrated + attn
        
        # Memory persistence logic [cite: 114]
        if self.role == 'memory':
            new_memory = 0.7 * self.hidden_memory + 0.3 * x
            self.hidden_memory = new_memory.detach()
            x = new_memory 

        return x + self.mlp(self.norm2(x))

class SpecializedGrid3x3(nn.Module):
    """
    A 3x3 Grid representing a 'super-neuron' assembly with defined topology[cite: 53].
    """
    def __init__(self, d_model, nhead):
        super(SpecializedGrid3x3, self).__init__()
        self.rows = 3
        self.cols = 3
        layout = [['memory', 'expert', 'memory'],
                  ['expert',  'router', 'expert'],
                  ['memory', 'expert', 'memory']]
        
        self.grid = nn.ModuleList([
            nn.ModuleList([SpecialistNode(d_model, nhead, layout[r][c]) for c in range(3)])
            for r in range(3)
        ])

    def get_neighbors(self, r, c, states):
        neighbors = []
        if r > 0: neighbors.append(states[r-1][c])
        if r < 2: neighbors.append(states[r+1][c])
        if c > 0: neighbors.append(states[r][c-1])
        if c < 2: neighbors.append(states[r][c+1])
        return neighbors

    def forward(self, seed, steps=3):
        batch = seed.size(0)
        d_model = seed.size(2)
        states = [[torch.zeros(batch, 1, d_model) for _ in range(3)] for _ in range(3)]
        states[1][1] = seed # Inject seed into router
        
        for _ in range(steps):
            next_states = [[None]*3 for _ in range(3)]
            for r in range(3):
                for c in range(3):
                    next_states[r][c] = self.grid[r][c](self.get_neighbors(r, c, states))
            states = next_states
        return states
