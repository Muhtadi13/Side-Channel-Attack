from flask import Flask, send_from_directory
# additional imports
from flask import request, jsonify, send_file
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os

app = Flask(__name__)

stored_traces = []
stored_heatmaps = []

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

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
    4. Return the heatmap image and optionally other statistics to the frontend
    """

    try:
        # Receive trace data
        data = request.get_json()
        if not data or 'trace' not in data:
            return jsonify({'error': 'No trace data provided'}), 400
        trace = data['trace']

        # print(trace)
        print(f"Received trace: {trace[:10]}...") 

        # Convert trace to a 2D array for heatmap (1 row, 1000 columns)
        trace_array = np.array(trace).reshape(1, -1)

        # Generate heatmap
        plt.figure(figsize=(10, 1))
        plt.imshow(trace_array, cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Sweep Count')
        plt.title('Trace Heatmap')
        # plt.xlabel('Time Window')
        plt.yticks([])  # No y-axis ticks for a single row

        # Calculate statistics
        trace_min = int(np.min(trace))
        trace_max = int(np.max(trace))
        trace_range = trace_max - trace_min
        samples = len(trace)

        # Add statistics as text below the plot
        plt.figtext(0.1, -0.2, f'Min: {trace_min}, Max: {trace_max}, Range: {trace_range}, Samples: {samples}')

        # Save plot to BytesIO
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)

        print("Heatmap generated successfully")

        # Store data
        stored_traces.append(trace)
        stored_heatmaps.append(img.getvalue())

        return send_file(
            io.BytesIO(img.getvalue()),
            mimetype='image/png',
            as_attachment=False
        )
    except Exception:
        return jsonify({'error': str(Exception)}), 500

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """ 
    Implment a clear results endpoint to reset stored data.
    1. Clear stored traces and heatmaps
    2. Return success/error message
    """
    try:
        stored_traces.clear()
        stored_heatmaps.clear()
        return jsonify({'message': 'Results cleared successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Additional endpoints can be implemented here as needed.

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)




def collect_fingerprints(driver, target_counts=None):
    """ Implement the main logic to collect fingerprints.
    1. Calculate the number of traces remaining for each website
    2. Open the fingerprinting website
    3. Collect traces for each website until the target number is reached
    4. Save the traces to the database
    5. Return the total number of new traces collected
    """

    wait = WebDriverWait(driver, 10)
    total_new_traces = 0
    trace_requests = []  # List to store website requests in order

    # Calculate remaining traces needed for each website
    current_counts = database.db.get_traces_collected()
    if target_counts is None:
        target_counts = {website: TRACES_PER_SITE for website in WEBSITES}

    remaining_counts = {website: max(0, target_counts.get(website, TRACES_PER_SITE) - current_counts.get(website, 0))
                       for website in WEBSITES}

    print("Remaining traces to collect:", remaining_counts)

    # Continue collecting until no traces remain
    while sum(remaining_counts.values()) > 0:
        for website in WEBSITES:
            if remaining_counts[website] <= 0:
                continue

            print(f"Collecting trace for {website} ({TRACES_PER_SITE - remaining_counts[website] + 1}/{TRACES_PER_SITE})")
            success = collect_single_trace(driver, wait, website)
            if success:
                total_new_traces += 1
                remaining_counts[website] -= 1
                trace_requests.append(website)  # Track the website for this trace
            else:
                print(f"  - Retrying trace collection for {website}")

            time.sleep(1)  # Brief pause to avoid overwhelming the server

    # After collecting all traces, retrieve them from the backend
    if total_new_traces > 0:
        print(f"Retrieving {total_new_traces} traces from backend...")
        traces = retrieve_traces_from_backend(driver)
        if not traces:
            print("  - Failed to retrieve traces from backend")
            return total_new_traces

        # Ensure the number of retrieved traces matches the number of successful collections
        if len(traces) != total_new_traces:
            print(f"  - Mismatch: Expected {total_new_traces} traces, but retrieved {len(traces)}")
            return total_new_traces

        # Pair each trace with its corresponding website
        collected_traces = []
        for i, trace in enumerate(traces):
            website = trace_requests[i]
            collected_traces.append({'website': website, 'trace_data': trace})

        # Save all traces to the database in one batch
        print(f"Saving {len(collected_traces)} traces to the database...")
        session = database.db.Session()
        try:
            for trace in collected_traces:
                website_url = trace['website']
                latest_trace = trace['trace_data']
                website_index = WEBSITES.index(website)  # Get website_index from WEBSITES list
                # Calculate website_index based on current count in database
                # website_index = session.query(func.count(database.Fingerprint.id)).filter(
                #     database.Fingerprint.website == website_url
                # ).scalar() or 0
                database.db.save_trace(website_url, website_index, latest_trace)

            session.commit()
            print("All traces saved successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error saving traces to database: {str(e)}")
        finally:
            session.close()

        # Export to JSON after saving all traces
        try:
            database.db.export_to_json(OUTPUT_PATH)
            print(f"Exported traces to {OUTPUT_PATH}")
        except Exception as e:
            print(f"Error exporting to JSON: {str(e)}")

    return total_new_traces
def set_website_index(self, website, new_index):
    """Set or update the website_index for a specific website's fingerprints."""
    session = self.Session()
    try:
        # Update all fingerprints for the given website
        updated_count = session.query(Fingerprint).filter_by(website=website).update({
            Fingerprint.website_index: new_index
        })
        session.commit()
        print(f"Updated website_index to {new_index} for {updated_count} records of {website}")
        return True
    except Exception as e:
        session.rollback()
        print(f"Error updating website_index: {str(e)}")
        return False
    finally:
        session.close()

