import os
import argparse
from golf_game import GolfGame
from agent import DQNAgent
from train import train, parse_args as train_args
from play import main as play_main, parse_args as play_args

def main():
    """
    Demo script to showcase the Golf AI project.
    
    This script demonstrates:
    1. Quick training of the agent
    2. Playing against the trained agent
    3. Watching the agent play against itself
    """
    print("=" * 50)
    print("Welcome to the Golf Card Game AI Demo!")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Train the agent (quick mode)")
        print("2. Train the agent (custom settings)")
        print("3. Play against the trained agent")
        print("4. Watch the agent play")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            # Quick training
            print("\nTraining the agent in quick mode (100 episodes)...")
            args = train_args()
            args.episodes = 100
            args.eval_interval = 25
            train(args)
            print("Quick training completed!")
            
        elif choice == "2":
            # Custom training
            print("\nTraining with custom settings...")
            args = train_args()
            
            # Get custom parameters
            try:
                args.episodes = int(input("Number of episodes (default: 10000): ") or "10000")
                args.batch_size = int(input("Batch size (default: 64): ") or "64")
                args.hidden_size = int(input("Hidden size (default: 128): ") or "128")
                args.lr = float(input("Learning rate (default: 0.001): ") or "0.001")
                args.eval_interval = int(input("Evaluation interval (default: 500): ") or "500")
            except ValueError:
                print("Invalid input. Using default values.")
                args = train_args()
            
            train(args)
            print("Custom training completed!")
            
        elif choice == "3":
            # Play against the agent
            print("\nPlaying against the trained agent...")
            args = play_args()
            args.mode = "play"
            
            # Check if model exists
            model_path = "models/best_model.pth"
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}. Using untrained agent.")
            
            play_main(args)
            
        elif choice == "4":
            # Watch the agent play
            print("\nWatching the agent play...")
            args = play_args()
            args.mode = "watch"
            
            # Check if model exists
            model_path = "models/best_model.pth"
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}. Using untrained agent.")
            
            play_main(args)
            
        elif choice == "5":
            # Exit
            print("\nThank you for trying the Golf Card Game AI!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 