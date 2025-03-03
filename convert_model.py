import argparse
import torch
from agent import DQNAgent

def main():
    parser = argparse.ArgumentParser(description='Convert a trained model to use the new one-hot encoded state representation')
    parser.add_argument('--input_model', type=str, required=True, help='Path to the input model file')
    parser.add_argument('--output_model', type=str, required=True, help='Path to save the converted model')
    args = parser.parse_args()
    
    print(f"Converting model from {args.input_model} to {args.output_model}...")
    
    # Create a new agent with the new state size (196)
    agent = DQNAgent(state_size=196)
    
    # Load the old model - this will trigger the conversion logic
    agent.load(args.input_model)
    
    # Save the converted model
    agent.save(args.output_model)
    
    print(f"Model successfully converted and saved to {args.output_model}")
    print("You can now use this model with the new one-hot encoded state representation.")

if __name__ == "__main__":
    main() 