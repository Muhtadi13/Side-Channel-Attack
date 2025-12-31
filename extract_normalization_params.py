import json
import numpy as np

DATASET_PATH = "dataset_2005013.json" 

def extract_min_max():
    # Load the dataset
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)

    # Extract all trace_data arrays
    traces = []
    for entry in data:
        trace = entry['trace_data']
        traces.append(trace)

    # Convert to NumPy array and compute min and max
    traces_array = np.array(traces, dtype=np.float32)
    min_val = np.min(traces_array)
    max_val = np.max(traces_array)

    # Save to normalization_params.json
    with open('normalization_params.json', 'w') as f:
        json.dump({'min_val': float(min_val), 'max_val': float(max_val)}, f)
        print(f"Saved normalization parameters: min_val={min_val}, max_val={max_val}")

if __name__ == "__main__":
    extract_min_max()