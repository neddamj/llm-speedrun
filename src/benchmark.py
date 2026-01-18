from train_llm import main
from time import time

experiment_name = 'baseline'
val_loss, train_time = main(experiment_name=experiment_name)
print(f"Returned validation loss: {val_loss:.4f}, training time: {train_time:.2f}s")