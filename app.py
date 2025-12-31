from flask import Flask, send_from_directory
from flask import request, jsonify, send_file, Response
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
import torch
from train import ComplexFingerprintClassifier,FingerprintClassifier, INPUT_SIZE ,HIDDEN_SIZE
from collect import WEBSITES 
import json 



# Load precomputed min and max values (assumed from train.py)
with open('normalization_params.json', 'r') as f:
    norm_params = json.load(f)
min_val, max_val = norm_params['min_val'], norm_params['max_val']

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexFingerprintClassifier(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=len(WEBSITES))  # Adjust num_classes
model.load_state_dict(torch.load('saved_models/ComplexCNN.pth')) 
model.eval()
model.to(device)


def normalize_trace(trace):
    """Normalize trace data using precomputed min and max values."""
    trace = np.array(trace, dtype=np.float32)
    if len(trace) < INPUT_SIZE:
        trace = np.pad(trace, (0, INPUT_SIZE - len(trace)), 'constant')
    elif len(trace) > INPUT_SIZE:
        trace = trace[:INPUT_SIZE]
    return (trace - min_val) / (max_val - min_val) if max_val != min_val else trace

def predict_website(trace):
    """Predict website using the loaded model."""
    trace_tensor = torch.FloatTensor(normalize_trace(trace)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(trace_tensor)
        pred = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(pred)
        confidence = pred[pred_idx]
    return WEBSITES[pred_idx], confidence





app = Flask(__name__)

stored_traces = []
stored_heatmaps = []  # Stores raw image bytes

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict')
def predict_page():
    return send_from_directory('static', 'predict.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    """ 
    Implement the collect_trace endpoint to receive trace data from the frontend and generate a heatmap.
    1. Receive trace data from the frontend as JSON
    2. Generate a heatmap using matplotlib
    3. Store the heatmap and trace data in the backend temporarily
    4. Return all heatmaps (new and previous) as Base64 strings in a JSON response
    """

    try:
        # Receive trace data
        data = request.get_json()
        if not data or 'trace' not in data:
            return jsonify({'error': 'No trace data provided'}), 400
        trace = data['trace']

        # print(f"Received trace: {trace[:10]}...") 

        # Convert trace to a 2D array for heatmap (1 row, 1000 columns)
        trace_array = np.array(trace).reshape(1, -1)

        # Generate heatmap
        plt.figure(figsize=(10, 1))
        plt.imshow(trace_array, cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Sweep Count')
        # plt.title('Trace Heatmap')
        plt.yticks([])  # No y-axis ticks for a single row

        # plt.figtext(0.1, -0.2, f'Min: {trace_min}, Max: {trace_max}, Range: {trace_range}, Samples: {samples}')

        # Save plot to BytesIO
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)

        print("Heatmap generated successfully")

        # Store data
        stored_traces.append(trace)
        stored_heatmaps.append(img.getvalue())

        # Convert all stored heatmaps to Base64
        all_heatmaps = []
        for i, (heatmap_data, trace_data) in enumerate(zip(stored_heatmaps, stored_traces)):
            heatmap_io = io.BytesIO(heatmap_data)
            base64_image = base64.b64encode(heatmap_io.getvalue()).decode('utf-8')
            # Calculate statistics for the current trace
            trace_min = int(np.min(trace_data))
            trace_max = int(np.max(trace_data))
            trace_range = trace_max - trace_min
            samples = len(trace_data)
            all_heatmaps.append({
                'url': f'data:image/png;base64,{base64_image}',
                'label': f'Min: {trace_min}, Max: {trace_max}, Range: {trace_range}, Samples: {samples}'
            })
        return jsonify({
            'status': 'success',
            'traces' : stored_traces,
            'heatmaps': all_heatmaps
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/traces', methods=['GET'])
def get_traces():
    try:
        return jsonify({
            'status': 'success',
            'traces': stored_traces
        }, 200)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/heatmaps', methods=['GET'])
def get_heatmaps():
    try:
        all_heatmaps = []
        for i, (heatmap_data, trace_data) in enumerate(zip(stored_heatmaps, stored_traces)):
            heatmap_io = io.BytesIO(heatmap_data)
            base64_image = base64.b64encode(heatmap_io.getvalue()).decode('utf-8')
            # Calculate statistics for the current trace
            trace_min = int(np.min(trace_data))
            trace_max = int(np.max(trace_data))
            trace_range = trace_max - trace_min
            samples = len(trace_data)
            all_heatmaps.append({
                'url': f'data:image/png;base64,{base64_image}',
                'label': f'Min: {trace_min}, Max: {trace_max}, Range: {trace_range}, Samples: {samples}'
            })
        return jsonify({
            'status': 'success',
            'heatmaps': all_heatmaps
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """ 
    Implement a clear results endpoint to reset stored data.
    1. Clear stored traces and heatmaps
    2. Return success/error message
    """
    try:
        stored_traces.clear()
        stored_heatmaps.clear()
        return jsonify({'message': 'Results cleared successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/get_results', methods=['GET'])
def get_results():
    try:
        # Assuming stored_traces contains the trace data
        return jsonify({'traces': stored_traces}), 200
    except Exception as e:
        print(f"Error in get_results: {str(e)}")
        return jsonify({'traces': [], 'error': str(e)}), 500

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        # print("Received data for prediction:", data)  # Debug print
        trace = data['trace']
        website, confidence = predict_website(trace)
        print(f"Predicted website: {website}, Confidence: {confidence}")  # Debug print
        # Convert confidence to native float
        return jsonify({'website': website, 'confidence': float(confidence)})
    except Exception as e:
        import traceback
        print("Error in /predict_api:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

