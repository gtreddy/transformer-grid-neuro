import torch
import torch.nn as nn

class RobotController(nn.Module):
    """
    A lightweight controller designed to inherit weights from the Grid[cite: 105].
    """
    def __init__(self, d_model):
        super(RobotController, self).__init__()
        self.processor = nn.Sequential(
            nn.Linear(d_model, d_model * 8), 
            nn.GELU(),
            nn.Linear(d_model * 8, d_model)
        )
        self.motor_output = nn.Linear(d_model, 4) 

    def forward(self, x):
        return self.motor_output(self.processor(x))

def transfer_knowledge(grid, robot, coords=(0, 1)):
    """
    Extracts weights from a Grid Expert and implants them into the Robot[cite: 91].
    """
    expert = grid.grid[coords[0]][coords[1]]
    print(f"Transferring knowledge from {expert.role.upper()} node at {coords}...")
    robot.processor.load_state_dict(expert.mlp.state_dict())
    print("Transfer Complete: Robot has inherited expertise.")
