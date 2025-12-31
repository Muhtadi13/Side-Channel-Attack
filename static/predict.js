function predictApp() {
  return {
    status: "",
    isCollecting: false,
    statusIsError: false,
    predictedWebsite: "Not detected",
    confidence: "0.0",

    // Collect a trace and get a prediction from the backend
    async collectAndPredict() {
      this.isCollecting = true;
      this.status = "Collecting trace data...";
      this.statusIsError = false;

      try {
        let worker = new Worker("worker.js");
        const trace = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });
        worker.terminate();

        this.status = "Predicting...";
        const response = await fetch(
          "/predict_api",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ trace })
          }
        );
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        this.predictedWebsite = data.website;
        this.confidence = (data.confidence * 100).toFixed(2);
        this.status = "Prediction updated!";
      } catch (error) {
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Start real-time prediction loop
    async startRealtimePrediction() {
      this.status = "Starting real-time prediction...";
      this.isCollecting = false;
      this.statusIsError = false;
      while (true) {
        await this.collectAndPredict();
        await new Promise((resolve) => setTimeout(resolve, 1000)); // 1s interval
      }
    },
  };
}

document.addEventListener("alpine:init", () => {
  Alpine.data("predictApp", predictApp);
});