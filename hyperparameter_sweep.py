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
    parser.add_argument('--report-interval', type=int, default=10,
                        help='Interval (in minutes) to write progress reports')
    parser.add_argument('--html-report', action='store_true',
                        help='Generate HTML report with interactive visualizations')
    return parser.parse_args()

def get_default_sweep_config():
    """Default hyperparameter configurations to sweep."""
    return {
        "lr": [0.0001, 0.0005, 0.001],
        "batch_size": [128, 256, 512],
        "hidden_size": [256, 512],
        "gamma": [0.95, 0.99],
        "epsilon_start": [1.0],
        "epsilon_end": [0.01, 0.05],
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
    
    # Save the command to a file for reference
    with open(os.path.join(run_dir, "command.txt"), 'w') as f:
        f.write(' '.join(cmd))
    
    start_time = time.time()
    
    try:
        # Run the process and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Open log files
        with open(os.path.join(run_dir, "output.log"), 'w') as stdout_log, \
             open(os.path.join(run_dir, "error.log"), 'w') as stderr_log:
            
            # Function to handle output streams
            def log_stream(stream, log_file, prefix=""):
                for line in stream:
                    log_file.write(f"{prefix}{line}")
                    log_file.flush()
                    # Also print to console for visibility
                    print(f"Run {run_id} {prefix}{line.rstrip()}")
            
            # Create threads to handle stdout and stderr
            import threading
            stdout_thread = threading.Thread(
                target=log_stream, 
                args=(process.stdout, stdout_log)
            )
            stderr_thread = threading.Thread(
                target=log_stream, 
                args=(process.stderr, stderr_log, "ERROR: ")
            )
            
            # Start threads
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Wait for logging to complete
            stdout_thread.join()
            stderr_thread.join()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check if process completed successfully
        if return_code == 0:
            print(f"Run {run_id} completed successfully in {duration:.2f} seconds")
            return True, run_id, duration
        else:
            print(f"Run {run_id} failed with return code {return_code}")
            
            # Add a summary to the error log
            with open(os.path.join(run_dir, "error.log"), 'a') as f:
                f.write(f"\n\nProcess exited with code {return_code}\n")
            
            return False, run_id, duration
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Run {run_id} failed with exception: {str(e)}")
        
        # Log the exception
        with open(os.path.join(run_dir, "error.log"), 'w') as error_file:
            error_file.write(f"Exception: {str(e)}\n")
            import traceback
            error_file.write(traceback.format_exc())
            
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
        for key in ['lr', 'batch_size', 'hidden_size', 'gamma', 
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
    params = ['lr', 'batch_size', 'hidden_size', 'gamma', 
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

def write_progress_report(output_dir, completed_runs, total_runs, start_time):
    """Write a progress report with completed runs and best configurations."""
    report_path = os.path.join(output_dir, "progress_report.md")
    
    # Get list of completed run directories
    run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_")]
    
    # Collect results from completed runs
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
            # Run completed but no metrics found
            results.append({
                'run_id': run_dir,
                'status': 'completed_no_metrics',
                **{k: v for k, v in config.items() if k != 'memory_estimate'}
            })
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
                'status': 'completed_with_metrics',
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
            # Run completed but error analyzing metrics
            results.append({
                'run_id': run_dir,
                'status': 'error_analyzing',
                'error': str(e),
                **{k: v for k, v in config.items() if k != 'memory_estimate'}
            })
    
    # Create DataFrame for analysis
    if results:
        results_df = pd.DataFrame(results)
        
        # Filter for runs with metrics
        metrics_df = results_df[results_df['status'] == 'completed_with_metrics'].copy()
        
        if not metrics_df.empty:
            # Sort by best score difference (primary) and best win rate (secondary)
            metrics_df = metrics_df.sort_values(
                by=['best_score_diff', 'best_win_rate'], 
                ascending=[False, False]
            )
            
            # Get top 5 configurations
            top5 = metrics_df.head(5)
        else:
            top5 = pd.DataFrame()
    else:
        results_df = pd.DataFrame()
        top5 = pd.DataFrame()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Write report
    with open(report_path, 'w') as f:
        f.write(f"# Hyperparameter Sweep Progress Report\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Progress\n\n")
        f.write(f"- Completed runs: {completed_runs}/{total_runs} ({completed_runs/total_runs*100:.1f}%)\n")
        f.write(f"- Elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
        
        if not top5.empty:
            f.write(f"\n## Top 5 Configurations So Far\n\n")
            
            for i, row in top5.iterrows():
                f.write(f"### Rank {i+1}: {row['run_id']}\n\n")
                f.write(f"- Best Score Diff: {row['best_score_diff']:.4f} (Episode {row['best_score_episode']})\n")
                f.write(f"- Best Win Rate: {row['best_win_rate']:.4f} (Episode {row['best_win_episode']})\n")
                f.write(f"- Final Win Rate: {row['final_win_rate']:.4f}\n")
                f.write(f"- Final Score Diff: {row['final_score_diff']:.4f}\n")
                f.write(f"- Configuration:\n")
                
                for key in ['lr', 'batch_size', 'hidden_size', 'gamma', 
                            'epsilon_decay', 'learn_every', 'target_update']:
                    if key in row:
                        f.write(f"  - {key}: {row[key]}\n")
                
                f.write("\n")
        
        # Write status counts
        if not results_df.empty:
            status_counts = results_df['status'].value_counts()
            f.write(f"\n## Run Status Summary\n\n")
            for status, count in status_counts.items():
                f.write(f"- {status}: {count}\n")
        
        # Write parameter distribution of completed runs
        if not results_df.empty:
            f.write(f"\n## Parameter Distribution of Completed Runs\n\n")
            
            for param in ['lr', 'batch_size', 'hidden_size', 'gamma', 
                          'epsilon_decay', 'learn_every', 'target_update']:
                if param in results_df.columns:
                    f.write(f"### {param}\n\n")
                    value_counts = results_df[param].value_counts().sort_index()
                    for value, count in value_counts.items():
                        f.write(f"- {value}: {count}\n")
                    f.write("\n")
    
    print(f"Progress report written to {report_path}")
    
    # Also save as CSV for easier analysis
    if not results_df.empty:
        csv_path = os.path.join(output_dir, "progress_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Progress results saved to {csv_path}")

def create_html_report(results_df, output_dir):
    """Create an HTML report with interactive visualizations."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Warning: plotly not installed. Skipping HTML report generation.")
        print("Install with: pip install plotly")
        return
    
    if results_df.empty:
        print("No results to generate HTML report.")
        return
    
    # Filter for runs with metrics
    metrics_df = results_df[results_df['status'] == 'completed_with_metrics'].copy()
    
    if metrics_df.empty:
        print("No runs with metrics to generate HTML report.")
        return
    
    # Create directory for HTML report
    html_dir = os.path.join(output_dir, "html_report")
    os.makedirs(html_dir, exist_ok=True)
    
    # Create main HTML file
    html_path = os.path.join(html_dir, "index.html")
    
    # Create parallel coordinates plot
    params = ['lr', 'batch_size', 'hidden_size', 'gamma', 
              'epsilon_decay', 'learn_every', 'target_update']
    metrics = ['best_score_diff', 'best_win_rate', 'final_win_rate']
    
    # Prepare data for parallel coordinates
    plot_df = metrics_df.copy()
    
    # Create color scale based on best_score_diff
    plot_df['color'] = plot_df['best_score_diff']
    
    # Create parallel coordinates plot
    fig_parallel = px.parallel_coordinates(
        plot_df,
        dimensions=params + metrics,
        color="color",
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Hyperparameter Parallel Coordinates Plot",
        labels={col: col for col in params + metrics}
    )
    
    # Save as HTML
    parallel_path = os.path.join(html_dir, "parallel_coordinates.html")
    fig_parallel.write_html(parallel_path)
    
    # Create scatter matrix
    fig_scatter = px.scatter_matrix(
        plot_df,
        dimensions=params + metrics,
        color="color",
        title="Hyperparameter Scatter Matrix",
        labels={col: col for col in params + metrics}
    )
    
    # Save as HTML
    scatter_path = os.path.join(html_dir, "scatter_matrix.html")
    fig_scatter.write_html(scatter_path)
    
    # Create bar charts for each parameter
    param_figs = {}
    
    for param in params:
        if param not in plot_df.columns:
            continue
            
        # Group by parameter and calculate mean and std of metrics
        grouped = plot_df.groupby(param)[metrics].agg(['mean', 'std']).reset_index()
        
        # Create subplots for each metric
        fig = make_subplots(rows=1, cols=len(metrics), 
                           subplot_titles=[f"Effect of {param} on {metric}" for metric in metrics])
        
        for i, metric in enumerate(metrics):
            mean_col = (metric, 'mean')
            std_col = (metric, 'std')
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=grouped[param],
                    y=grouped[mean_col],
                    error_y=dict(
                        type='data',
                        array=grouped[std_col],
                        visible=True
                    ),
                    name=metric
                ),
                row=1, col=i+1
            )
            
            # Update layout
            fig.update_xaxes(title_text=param, row=1, col=i+1)
            fig.update_yaxes(title_text=metric, row=1, col=i+1)
        
        # Update overall layout
        fig.update_layout(
            title=f"Effect of {param} on Performance Metrics",
            height=500,
            width=1200
        )
        
        # Save as HTML
        param_path = os.path.join(html_dir, f"param_{param}.html")
        fig.write_html(param_path)
        
        param_figs[param] = f"param_{param}.html"
    
    # Create 3D scatter plot
    fig_3d = px.scatter_3d(
        plot_df,
        x='lr',
        y='batch_size',
        z='hidden_size',
        color='best_score_diff',
        size='best_win_rate',
        hover_data=params + metrics,
        title="3D Visualization of Key Hyperparameters",
        labels={col: col for col in params + metrics}
    )
    
    # Save as HTML
    scatter_3d_path = os.path.join(html_dir, "scatter_3d.html")
    fig_3d.write_html(scatter_3d_path)
    
    # Create main HTML file with links to all plots
    with open(html_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hyperparameter Sweep Results</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }
                h1, h2, h3 {
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .plot-container {
                    margin: 20px 0;
                }
                .nav-links {
                    margin: 20px 0;
                    padding: 10px;
                    background-color: #f2f2f2;
                    border-radius: 5px;
                }
                .nav-links a {
                    margin-right: 15px;
                    text-decoration: none;
                    color: #0066cc;
                }
                .nav-links a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Hyperparameter Sweep Results</h1>
                <p>Generated at: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                
                <div class="nav-links">
                    <a href="#overview">Overview</a>
                    <a href="#top-configs">Top Configurations</a>
                    <a href="#visualizations">Visualizations</a>
                    <a href="#parameter-effects">Parameter Effects</a>
                </div>
                
                <h2 id="overview">Overview</h2>
                <p>Total runs analyzed: """ + str(len(metrics_df)) + """</p>
                
                <h2 id="top-configs">Top Configurations</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Run ID</th>
                        <th>Best Score Diff</th>
                        <th>Best Win Rate</th>
                        <th>Final Win Rate</th>
                        <th>Final Score Diff</th>
                        """ + ''.join([f"<th>{param}</th>" for param in params]) + """
                    </tr>
        """)
        
        # Add top 10 configurations
        top10 = metrics_df.sort_values(
            by=['best_score_diff', 'best_win_rate'], 
            ascending=[False, False]
        ).head(10)
        
        for i, row in top10.iterrows():
            f.write(f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{row['run_id']}</td>
                        <td>{row['best_score_diff']:.4f}</td>
                        <td>{row['best_win_rate']:.4f}</td>
                        <td>{row['final_win_rate']:.4f}</td>
                        <td>{row['final_score_diff']:.4f}</td>
            """)
            
            for param in params:
                if param in row:
                    f.write(f"<td>{row[param]}</td>")
                else:
                    f.write("<td>-</td>")
            
            f.write("</tr>")
        
        f.write("""
                </table>
                
                <h2 id="visualizations">Visualizations</h2>
                
                <h3>Interactive Plots</h3>
                <ul>
                    <li><a href="parallel_coordinates.html" target="_blank">Parallel Coordinates Plot</a></li>
                    <li><a href="scatter_matrix.html" target="_blank">Scatter Matrix</a></li>
                    <li><a href="scatter_3d.html" target="_blank">3D Scatter Plot</a></li>
                </ul>
                
                <h2 id="parameter-effects">Parameter Effects</h2>
                <ul>
        """)
        
        # Add links to parameter effect plots
        for param, html_file in param_figs.items():
            f.write(f'<li><a href="{html_file}" target="_blank">Effect of {param}</a></li>')
        
        f.write("""
                </ul>
            </div>
        </body>
        </html>
        """)
    
    print(f"HTML report generated at {html_path}")
    return html_path

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
    
    # Track start time and completed runs
    start_time = time.time()
    completed_runs = 0
    last_report_time = start_time
    
    # Run all batches
    start_id = 0
    for i, batch in enumerate(batches):
        print(f"\nRunning batch {i+1}/{len(batches)} with {len(batch)} configurations")
        run_batch(batch, output_dir, start_id)
        
        # Update completed runs count
        completed_runs += len(batch)
        start_id += len(batch)
        
        # Check if it's time to write a progress report
        current_time = time.time()
        if (current_time - last_report_time) >= (args.report_interval * 60):
            write_progress_report(output_dir, completed_runs, total_runs, start_time)
            last_report_time = current_time
    
    # Write final progress report
    write_progress_report(output_dir, completed_runs, total_runs, start_time)
    
    # Analyze results
    results_df = analyze_results(output_dir)
    
    # Create script with best configuration
    if results_df is not None and not results_df.empty:
        create_best_config_script(results_df, output_dir)
        
        # Generate HTML report if requested
        if args.html_report:
            create_html_report(results_df, output_dir)
    
    print("\nHyperparameter sweep complete!")

if __name__ == "__main__":
    # Set the start method for multiprocessing to 'spawn' for better CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main() 