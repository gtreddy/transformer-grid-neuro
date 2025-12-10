import torch
from grid_system import SpecializedGrid3x3
from robot_transfer import RobotController, transfer_knowledge

def main():
    print("--- Neuronal Transformer Grid Simulation ---")
    
    # 1. Initialize
    D_MODEL = 16
    grid = SpecializedGrid3x3(D_MODEL, nhead=2)
    seed = torch.randn(1, 1, D_MODEL)
    
    # 2. Simulate Refinement [cite: 62]
    print("Running Grid Simulation...")
    grid(seed, steps=5)
    
    # 3. Robotic Transfer [cite: 105]
    robot = RobotController(D_MODEL)
    transfer_knowledge(grid, robot, expert_coords=(0, 1))

if __name__ == "__main__":
    main()
