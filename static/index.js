function app() {
  return {
    /* This is the main app object containing all the application state and methods. */
    // The following properties are used to store the state of the application

    // results of cache latency measurements
    latencyResults: null,
    // local collection of trace data
    traceData: [],
    // Local collection of heapmap images
    heatmaps: [],

    // Current status message
    status: "",
    // Is any worker running?
    isCollecting: false,
    // Is the status message an error?
    statusIsError: false,
    // Show trace data in the UI?
    showingTraces: false,

    // Collect latency data using warmup.js worker
    async collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyResults = null;
      this.statusIsError = false;
      this.showingTraces = false;

      try {
        // Create a worker
        let worker = new Worker("warmup.js");

        // Start the measurement and wait for result
        const results = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        // Update results
        this.latencyResults = results;
        this.status = "Latency data collection complete!";

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting latency data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
        // worker.terminate();
        // ekhane terminate korenai . might be a problem
      }
    },

    // Collect trace data using worker.js and send to backend
    async collectTraceData() {
       /* 
        * Implement this function to collect trace data.
        * 1. Create a worker to run the sweep function.
        * 2. Collect the trace data from the worker.
        * 3. Send the trace data to the backend for temporary storage and heatmap generation.
        * 4. Fetch the heatmap from the backend and add it to the local collection.
        * 5. Handle errors and update the status.
        */

        this.isCollecting = true; 
        this.status = "Collecting trace data...";
        this.traceData = [];
        this.heatmaps = [];
        this.statusIsError = false;
        this.showingTraces = false;

        try{
          let worker = new Worker("worker.js");
          
          const tData = await new Promise((resolve)=>{
            worker.onmessage = (e) => {
              // console.log("Trace data received:", e.data);
              resolve(e.data);
            }
            worker.postMessage("start");
          });
          // for(let i =0 ;i<this.traceData.length;i++){
          //   console.log("trace data :\n" + this.traceData[i].length)
          // }
          const response = await fetch('/collect_trace' , {
            method : 'POST',
            headers : {
              'Content-Type' : 'application/json'
            },
            body : JSON.stringify({trace : tData})
          });
          
          if(!response.ok){
            console.error("Response error:", await response.text());
            throw new Error('Trace Data Sending Failed');
          }

          const data = await response.json();
          if (data.error) throw new Error(data.error);
          this.traceData = data.traces;
          console.log("trace data :\n" + this.traceData.length);
          
          this.heatmaps = data.heatmaps.map((hm, index) => ({
              url: hm.url,
              label: hm.label
          }));
          console.log("Heatmaps array:", this.heatmaps);

          this.status = "Trace data collection complete!";
          this.showingTraces = true;

          worker.terminate();
      

        }catch(error){

          console.error("Error collecting trace data:", error);
          this.status = `Error: ${error.message}`;
          this.statusIsError = true;

        }finally{

          this.isCollecting = false;
        }


    },

    // Download the trace data as JSON (array of arrays format for ML)
    async downloadTraces() {
       /* 
        * Implement this function to download the trace data.
        * 1. Fetch the latest data from the backend API.
        * 2. Create a download file with the trace data in JSON format.
        * 3. Handle errors and update the status.
        */
      this.isCollecting = true;
      this.status = "Downloading traces...";
      this.statusIsError = false;

      try {
        const response = await fetch('/api/traces', {
            method: 'GET'
        });
        if (!response.ok) throw new Error('Failed to fetch traces');
        
        const data = await response.json();
        const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const doc = document.createElement('a');
        doc.href = url;
        doc.download = 'traces.json';
        doc.click();
        URL.revokeObjectURL(url);
        this.status = "Traces downloaded!";
      } catch (error) {
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Clear all results from the server
    async clearResults() {
      /* 
       * Implement this function to clear all results from the server.
       * 1. Send a request to the backend API to clear all results.
       * 2. Clear local copies of trace data and heatmaps.
       * 3. Handle errors and update the status.
       */
      this.isCollecting = true;
      this.status = "Clearing results...";
      this.statusIsError = false;
      this.showingTraces = false;

      try {
          const response = await fetch('/api/clear_results', {
              method: 'POST'
          });
          if (!response.ok) throw new Error('Failed to clear results');
          
          this.traceData = [];
          this.heatmaps = [];
          this.status = "Cleared all results!";
      } catch (error) {
          this.status = `Error: ${error.message}`;
          this.statusIsError = true;
      } finally {
          this.isCollecting = false;
      }
    },


    async fetchResults() {
      this.status = "Fetching previous heatmaps...";
      this.statusIsError = false;

      try {
          console.log("Fetching heatmaps...");  
          const response = await fetch('/api/heatmaps');
          console.log("Response status:", response.status); 
          if (!response.ok) {
              throw new Error(`Failed to fetch heatmaps: ${response.statusText}`);
          }
          const data = await response.json();
          console.log("Fetched data:", data);
          if (data.error) {
              throw new Error(`Backend error: ${data.error}`);
          }
          this.heatmaps = data.heatmaps || [];
          this.showingTraces = this.heatmaps.length > 0;
          this.status = "Heatmaps loaded.";
          console.log("Heatmaps updated:", this.heatmaps);  // Log the updated heatmaps
          console.log(this.showingTraces);
          // console.log
      } catch (error) {
          console.error("Error in fetchResults:", error);  // Log any errors that occur
          this.status = `Error: ${error.message}`;
          this.statusIsError = true;
      }
    }


  };
}


