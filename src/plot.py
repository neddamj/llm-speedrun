import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_all_training_runs():
    """Load all training logs and plot validation loss curves."""
    
    # Find all training log files
    log_files = sorted(glob.glob("../logs/training_log_*.csv"))
    
    if not log_files:
        print("No training log files found!")
        print("Looking for files matching pattern: training_log_*.csv")
        return
    
    print(f"Found {len(log_files)} training log files")
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot each log file
    for log_file in log_files:
        # Extract timestamp from filename
        timestamp = log_file.replace("training_log_", "").replace(".csv", "")
        
        # Read the CSV
        try:
            df = pd.read_csv(log_file)
            
            # Filter out rows where val_loss is 'N/A' or missing
            df_valid = df[df['val_loss'] != 'N/A'].copy()
            df_valid['val_loss'] = pd.to_numeric(df_valid['val_loss'], errors='coerce')
            df_valid = df_valid.dropna(subset=['val_loss'])
            
            if len(df_valid) > 0:
                # Plot the validation loss
                plt.plot(df_valid['step'], df_valid['val_loss'], 
                        marker='o', label=timestamp, linewidth=2, markersize=4)
                
                print(f"  {timestamp}: {len(df_valid)} validation points, "
                      f"final val_loss = {df_valid['val_loss'].iloc[-1]:.4f}")
            else:
                print(f"  {timestamp}: No valid validation data")
                
        except Exception as e:
            print(f"  Error reading {log_file}: {e}")
    
    # Customize the plot
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss Across Training Runs', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('../figs', exist_ok=True)
    output_file = '../figs/training_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    print("=" * 80)
    print("Training Log Visualization")
    print("=" * 80)
    
    # Plot all runs
    plot_all_training_runs()