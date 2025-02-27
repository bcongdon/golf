#!/usr/bin/env python3
import os
import sys
import argparse
import itertools
import multiprocessing
import subprocess
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for Golf Card Game AI')
    parser.add_argument('--max-parallel', type=int, default=3, 
                        help='Maximum number of parallel training runs')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of episodes per training run')
    parser.add_argument('--eval-interval', type=int, default=250,
                        help='Evaluation interval for each run')
    parser.add_argument('--output-dir', type=str, default='sweep_results',
                        help='Directory to save sweep results')
    parser.add_argument('--gpu-memory', type=int, default=8000,
                        help='Available GPU memory in MB')
    parser.add_argument('--sweep-config', type=str, default=None,
                        help='Path to JSON file with sweep configuration')
    return parser.parse_args()

def get_default_sweep_config():
    """Default hyperparameter configurations to sweep."""
    return {
        "learning_rate": [0.0001, 0.0005, 0.001],
        "batch_size": [128, 256, 512],
        "hidden_size": [256, 512],
        "gamma": [0.95, 0.99],
        "epsilon_decay": [0.9995, 0.9998],
        "learn_every": [1, 4, 8],
        "target_update": [500, 1000, 2000]
    }

def load_sweep_config(config_path):
    """Load sweep configuration from JSON file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return get_default_sweep_config()

def estimate_memory_usage(batch_size, hidden_size):
    """Estimate GPU memory usage in MB for a configuration."""
    # This is a rough estimate - adjust based on your model's actual memory usage
    base_memory = 500  # Base memory usage in MB
    batch_factor = batch_size / 128  # Relative to batch size 128
    hidden_factor = (hidden_size / 256) ** 2  # Quadratic scaling with hidden size
    
    return int(base_memory * batch_factor * hidden_factor)

def generate_run_configs(sweep_config, max_parallel, gpu_memory, episodes, eval_interval):
    """Generate all possible hyperparameter configurations."""
    # Get all possible combinations of hyperparameters
    keys = sweep_config.keys()
    values = sweep_config.values()
    combinations = list(itertools.product(*values))
    
    # Create a list of configurations
    configs = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        
        # Add fixed parameters
        config['episodes'] = episodes
        config['eval_interval'] = eval_interval
        
        # Estimate memory usage
        config['memory_estimate'] = estimate_memory_usage(
            config.get('batch_size', 256),
            config.get('hidden_size', 512)
        )
        
        configs.append(config)
    
    # Sort configurations by estimated memory usage (descending)
    # This helps pack configurations efficiently
    configs.sort(key=lambda x: x['memory_estimate'], reverse=True)
    
    # Group configurations into batches that can run in parallel
    batches = []
    current_batch = []
    current_memory = 0
    
    for config in configs:
        # If adding this config exceeds memory or max parallel limit, start a new batch
        if (current_memory + config['memory_estimate'] > gpu_memory or 
            len(current_batch) >= max_parallel):
            batches.append(current_batch)
            current_batch = []
            current_memory = 0
        
        current_batch.append(config)
        current_memory += config['memory_estimate']
    
    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches

def run_training(config, output_dir, run_id):
    """Run a single training configuration."""
    # Create a unique directory for this run
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    logs_dir = os.path.join(run_dir, "logs")
    models_dir = os.path.join(run_dir, "models")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Build command with all hyperparameters
    cmd = [sys.executable, "train.py"]
    
    # Add all parameters from config
    for key, value in config.items():
        # Skip non-command line parameters
        if key in ['memory_estimate']:
            continue
        
        # Convert parameter name from snake_case to --kebab-case
        param_name = f"--{key.replace('_', '-')}"
        cmd.extend([param_name, str(value)])
    
    # Add output directories
    cmd.extend(["--logs-dir", logs_dir])
    cmd.extend(["--save-dir", models_dir])
    
    # Add mixed precision for faster training
    cmd.append("--mixed-precision")
    
    # Run the training process
    print(f"Starting run {run_id} with config: {config}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Run the process and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output to a log file
        with open(os.path.join(run_dir, "output.log"), 'w') as log_file:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
        
        # Wait for process to complete
        process.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check if process completed successfully
        if process.returncode == 0:
            print(f"Run {run_id} completed successfully in {duration:.2f} seconds")
            return True, run_id, duration
        else:
            print(f"Run {run_id} failed with return code {process.returncode}")
            # Capture error output
            error_output = process.stderr.read()
            with open(os.path.join(run_dir, "error.log"), 'w') as error_file:
                error_file.write(error_output)
            
            return False, run_id, duration
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Run {run_id} failed with exception: {str(e)}")
        return False, run_id, duration

def run_batch(batch, output_dir, start_id):
    """Run a batch of configurations in parallel."""
    processes = []
    
    for i, config in enumerate(batch):
        run_id = start_id + i
        p = multiprocessing.Process(
            target=run_training,
            args=(config, output_dir, run_id)
        )
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

def analyze_results(output_dir):
    """Analyze and compare results from all runs."""
    print("\nAnalyzing results...")
    
    # Find all run directories
    run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_")]
    
    results = []
    
    for run_dir in run_dirs:
        run_path = os.path.join(output_dir, run_dir)
        
        # Load configuration
        config_path = os.path.join(run_path, "config.json")
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Find evaluation metrics
        logs_dir = os.path.join(run_path, "logs")
        eval_metrics_path = os.path.join(logs_dir, "eval_metrics.csv")
        
        if not os.path.exists(eval_metrics_path):
            print(f"No evaluation metrics found for {run_dir}")
            continue
        
        try:
            # Load metrics
            metrics_df = pd.read_csv(eval_metrics_path)
            
            # Get the best win rate and corresponding episode
            best_win_idx = metrics_df['WinRate'].idxmax()
            best_win_rate = metrics_df.loc[best_win_idx, 'WinRate']
            best_win_episode = metrics_df.loc[best_win_idx, 'Episode']
            
            # Get the best score difference and corresponding episode
            if 'ScoreDiff' in metrics_df.columns:
                best_score_idx = metrics_df['ScoreDiff'].idxmax()
                best_score_diff = metrics_df.loc[best_score_idx, 'ScoreDiff']
                best_score_episode = metrics_df.loc[best_score_idx, 'Episode']
            else:
                best_score_diff = float('nan')
                best_score_episode = float('nan')
            
            # Get the final metrics
            final_idx = metrics_df['Episode'].idxmax()
            final_win_rate = metrics_df.loc[final_idx, 'WinRate']
            final_score_diff = metrics_df.loc[final_idx, 'ScoreDiff'] if 'ScoreDiff' in metrics_df.columns else float('nan')
            
            # Add to results
            result = {
                'run_id': run_dir,
                'best_win_rate': best_win_rate,
                'best_win_episode': best_win_episode,
                'best_score_diff': best_score_diff,
                'best_score_episode': best_score_episode,
                'final_win_rate': final_win_rate,
                'final_score_diff': final_score_diff
            }
            
            # Add configuration parameters
            for key, value in config.items():
                if key != 'memory_estimate':
                    result[key] = value
            
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {run_dir}: {str(e)}")
    
    if not results:
        print("No valid results found to analyze")
        return
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Sort by best score difference (primary) and best win rate (secondary)
    results_df = results_df.sort_values(
        by=['best_score_diff', 'best_win_rate'], 
        ascending=[False, False]
    )
    
    # Save results to CSV
    results_path = os.path.join(output_dir, "sweep_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Print top 5 configurations
    print("\nTop 5 configurations:")
    top5 = results_df.head(5)
    for i, row in top5.iterrows():
        print(f"\nRank {i+1}:")
        print(f"  Run ID: {row['run_id']}")
        print(f"  Best Score Diff: {row['best_score_diff']:.4f} (Episode {row['best_score_episode']})")
        print(f"  Best Win Rate: {row['best_win_rate']:.4f} (Episode {row['best_win_episode']})")
        print(f"  Final Win Rate: {row['final_win_rate']:.4f}")
        print(f"  Final Score Diff: {row['final_score_diff']:.4f}")
        print("  Configuration:")
        for key in ['learning_rate', 'batch_size', 'hidden_size', 'gamma', 
                    'epsilon_decay', 'learn_every', 'target_update']:
            if key in row:
                print(f"    {key}: {row[key]}")
    
    # Create visualization of parameter importance
    create_parameter_importance_plots(results_df, output_dir)
    
    return results_df

def create_parameter_importance_plots(results_df, output_dir):
    """Create plots showing the impact of different hyperparameters."""
    print("\nCreating parameter importance plots...")
    
    # Parameters to analyze
    params = ['learning_rate', 'batch_size', 'hidden_size', 'gamma', 
              'epsilon_decay', 'learn_every', 'target_update']
    
    # Metrics to analyze
    metrics = ['best_score_diff', 'best_win_rate', 'final_win_rate']
    
    # Create a directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create plots for each parameter
    for param in params:
        if param not in results_df.columns:
            continue
            
        plt.figure(figsize=(12, 8))
        
        for i, metric in enumerate(metrics):
            if metric not in results_df.columns:
                continue
                
            plt.subplot(1, len(metrics), i+1)
            
            # Group by parameter and calculate mean and std of the metric
            grouped = results_df.groupby(param)[metric].agg(['mean', 'std']).reset_index()
            
            # Sort by parameter value
            grouped = grouped.sort_values(by=param)
            
            # Plot
            plt.errorbar(
                grouped[param], 
                grouped['mean'], 
                yerr=grouped['std'],
                marker='o',
                linestyle='-',
                capsize=5
            )
            
            plt.title(f'Effect of {param} on {metric}')
            plt.xlabel(param)
            plt.ylabel(metric)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels
            for x, y, std in zip(grouped[param], grouped['mean'], grouped['std']):
                plt.annotate(
                    f'{y:.3f}±{std:.3f}',
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'param_{param}.png'))
        plt.close()
    
    # Create a correlation heatmap
    plt.figure(figsize=(12, 10))
    
    # Select only numeric columns
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    corr_matrix = results_df[numeric_cols].corr()
    
    # Plot heatmap
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
    plt.colorbar(label='Correlation coefficient')
    plt.title('Correlation between parameters and metrics')
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.index)):
            plt.text(
                i, j, 
                f'{corr_matrix.iloc[j, i]:.2f}',
                ha='center', 
                va='center',
                color='white' if abs(corr_matrix.iloc[j, i]) > 0.5 else 'black'
            )
    
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()
    
    print(f"Plots saved to {plots_dir}")

def create_best_config_script(results_df, output_dir):
    """Create a script to run training with the best configuration."""
    if results_df.empty:
        return
    
    # Get the best configuration
    best_config = results_df.iloc[0].to_dict()
    
    # Create script content
    script_content = "#!/bin/bash\n\n"
    script_content += "# This script runs training with the best hyperparameters found during the sweep\n\n"
    
    # Add command with all parameters
    script_content += "python train.py \\\n"
    
    # Add all parameters from best config
    for key, value in best_config.items():
        # Skip non-command line parameters and metrics
        if key in ['memory_estimate', 'run_id', 'best_win_rate', 'best_win_episode', 
                   'best_score_diff', 'best_score_episode', 'final_win_rate', 'final_score_diff']:
            continue
        
        # Convert parameter name from snake_case to --kebab-case
        param_name = f"--{key.replace('_', '-')}"
        script_content += f"    {param_name} {value} \\\n"
    
    # Add mixed precision for faster training
    script_content += "    --mixed-precision \\\n"
    
    # Add output directories
    script_content += "    --logs-dir logs/best_config \\\n"
    script_content += "    --save-dir models/best_config\n"
    
    # Save script
    script_path = os.path.join(output_dir, "run_best_config.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"\nBest configuration script saved to {script_path}")
    print("Run this script to train with the best hyperparameters found during the sweep.")

def main():
    args = parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"sweep_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Hyperparameter sweep results will be saved to {output_dir}")
    
    # Load sweep configuration
    sweep_config = load_sweep_config(args.sweep_config)
    
    # Save the sweep configuration
    with open(os.path.join(output_dir, "sweep_config.json"), 'w') as f:
        json.dump(sweep_config, f, indent=2)
    
    # Generate run configurations
    batches = generate_run_configs(
        sweep_config, 
        args.max_parallel, 
        args.gpu_memory, 
        args.episodes, 
        args.eval_interval
    )
    
    total_runs = sum(len(batch) for batch in batches)
    print(f"Generated {total_runs} configurations in {len(batches)} batches")
    
    # Run all batches
    start_id = 0
    for i, batch in enumerate(batches):
        print(f"\nRunning batch {i+1}/{len(batches)} with {len(batch)} configurations")
        run_batch(batch, output_dir, start_id)
        start_id += len(batch)
    
    # Analyze results
    results_df = analyze_results(output_dir)
    
    # Create script with best configuration
    if results_df is not None and not results_df.empty:
        create_best_config_script(results_df, output_dir)
    
    print("\nHyperparameter sweep complete!")

if __name__ == "__main__":
    # Set the start method for multiprocessing to 'spawn' for better CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main() 