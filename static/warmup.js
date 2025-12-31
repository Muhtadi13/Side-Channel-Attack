/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;

function readNlines(n) {
  /*
   * Implement this function to read n cache lines.
   * 1. Allocate a buffer of size n * LINESIZE.
   * 2. Read each cache line (read the buffer in steps of LINESIZE) 10 times.
   * 3. Collect total time taken in an array using `performance.now()`.
   * 4. Return the median of the time taken in milliseconds.
   */

  const buffer = new ArrayBuffer(n * LINESIZE);
  const bufferWindow = new Uint8Array(buffer);

  const intervals = [];
  const cnt = 10;  

  for(let i=0;i<cnt;i++){
    const start=performance.now();
    for(let j=0;j<n*LINESIZE;j+=LINESIZE){
      bufferWindow[j];
    }
    const end=performance.now();
    intervals.push(end-start);
  }

  intervals.sort((a, b) => a - b);
  const medianIndex = Math.floor(intervals.length / 2);
  const median = (intervals.length % 2 === 1 ? intervals[medianIndex] : ( intervals[medianIndex]+intervals[medianIndex-1] )/2 );

  return median;
}

self.addEventListener("message", function (e) {
  if (e.data === "start") {
    const results = {};

    /* Call the readNlines function for n = 1, 10, ... 10,000,000 and store the result */

    for(let n=1;n<=100_000_000;n*=10){

      try{
        results[n]=readNlines(n);
      }catch(error){
        console.error(`Error for n=${n}:`, error);
        break; // Stop if an error occurs (e.g., buffer too large)
      }
    }
    self.postMessage(results);
  }
});
