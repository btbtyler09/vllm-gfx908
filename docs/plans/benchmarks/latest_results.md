# Testing the latests changes to the gptq kernel
The following tests were performed with changes to the GPTQ kernel.

## Changes to baseline vLLM:
1. nlzy's ockl_fdot2 optimization.
2. larger block sizes: BLOCK_KN_SIZE 256, MAX_Q_GEMM_ROWS 64, MAX_Q_GEMM_ROWS_8BIT 32

## Qwen 3 32B 4bit gptq
* vLLM 0.9.2 + our kernel modifications
* Power limit 290w (only tests 6 concurrency)
* -tp 4
* Model: kaitchup/Qwen3-32B-autoround-4bit-gptq
* max model len 8192

* Maximum request concurrency: 2


* Maximum request concurrency: 6
============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  53.13     
Total input tokens:                      51200     
Total generated tokens:                  5981      
Request throughput (req/s):              0.94      
Output token throughput (tok/s):         112.57    
Total Token throughput (tok/s):          1076.26   
---------------Time to First Token----------------
Mean TTFT (ms):                          165.37    
Median TTFT (ms):                        135.99    
P99 TTFT (ms):                           254.81    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          50.90     
Median TPOT (ms):                        51.09     
P99 TPOT (ms):                           51.75     
---------------Inter-token Latency----------------
Mean ITL (ms):                           50.91     
Median ITL (ms):                         50.42     
P99 ITL (ms):                            74.36     
==================================================

* vLLM 0.9.2 + our kernel modifications
* Power limit 225w (testing lower power limit, no throttling at this power)
* -tp 4
* Model: kaitchup/Qwen3-32B-autoround-4bit-gptq
* max model len 8192

* Maximum request concurrency: 2
============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  106.48    
Total input tokens:                      51200     
Total generated tokens:                  5989      
Request throughput (req/s):              0.47      
Output token throughput (tok/s):         56.25     
Total Token throughput (tok/s):          537.09    
---------------Time to First Token----------------
Mean TTFT (ms):                          111.72    
Median TTFT (ms):                        109.67    
P99 TTFT (ms):                           149.50    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          34.48     
Median TPOT (ms):                        34.45     
P99 TPOT (ms):                           37.66     
---------------Inter-token Latency----------------
Mean ITL (ms):                           34.38     
Median ITL (ms):                         33.75     
P99 ITL (ms):                            51.21     
==================================================

* Maximum request concurrency: 6
============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  53.34     
Total input tokens:                      51200     
Total generated tokens:                  5981      
Request throughput (req/s):              0.94      
Output token throughput (tok/s):         112.12    
Total Token throughput (tok/s):          1071.95   
---------------Time to First Token----------------
Mean TTFT (ms):                          171.96    
Median TTFT (ms):                        146.41    
P99 TTFT (ms):                           259.02    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          51.07     
Median TPOT (ms):                        51.21     
P99 TPOT (ms):                           52.34     
---------------Inter-token Latency----------------
Mean ITL (ms):                           51.08     
Median ITL (ms):                         50.48     
P99 ITL (ms):                            75.32     
==================================================

## Qwen 3 0.6B 8bit gptq
* vLLM 0.9.2 + our kernel modifications
* Power limit 225w (Running at 225watts from now on.)
* -tp 4
* Model: Qwen/Qwen3-0.6B-GPTQ-Int8
* max model len 32768

* Maximum request concurrency: 2
============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  43.36     
Total input tokens:                      51200     
Total generated tokens:                  5769      
Request throughput (req/s):              1.15      
Output token throughput (tok/s):         133.05    
Total Token throughput (tok/s):          1313.92   
---------------Time to First Token----------------
Mean TTFT (ms):                          41.05     
Median TTFT (ms):                        39.73     
P99 TTFT (ms):                           59.38     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.58     
Median TPOT (ms):                        14.50     
P99 TPOT (ms):                           16.63     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.51     
Median ITL (ms):                         14.21     
P99 ITL (ms):                            24.58     
==================================================

* Maximum request concurrency: 6
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  49.89     
Total input tokens:                      102400    
Total generated tokens:                  11661     
Request throughput (req/s):              2.00      
Output token throughput (tok/s):         233.73    
Total Token throughput (tok/s):          2286.23   
---------------Time to First Token----------------
Mean TTFT (ms):                          53.54     
Median TTFT (ms):                        52.82     
P99 TTFT (ms):                           67.36     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.71     
Median TPOT (ms):                        24.76     
P99 TPOT (ms):                           26.27     
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.69     
Median ITL (ms):                         24.50     
P99 ITL (ms):                            31.96     
==================================================

