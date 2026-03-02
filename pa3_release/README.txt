Last name of Student 1:
First name of Student 1:
Email of Student 1:
GradeScope account name of Student 1: 
Last name of Student 2:
First name of Student 2:
Email of Student 2:
GradeScope account name of Student 2: 

----------------------------------------------------------------------------
Report for Question 1 

List your code change for this question 

Implemented in mult_vec:
Compute linear thread id idx = blockIdx.x * blockDim.x + threadIdx.x;
Each thread updates its assigned rows y[i] = d[i] + sum A[i,j] * x[j]
Get elementwise difference diff[i] = |y[i] − x[i]|

Implemented in it_mult_vec:
Allocate/copy device memory using cudaMalloc and cudaMemcpy
Launch kernel each iteration
Swap x and y between iterations
Copy final result back


Parallel time for n=4K, t=1K,  4x128  threads
4.102072 s

Parallel time for n=4K, t=1K,  8x128  threads
2.047910 s

Parallel time for n=4K, t=1K,  16x128 threads
1.033989 s

Parallel time for n=4K, t=1K,  32x128 threads
0.560899 s

Do you see a trend of  speedup improvement  with more threads? We expect a good speedup and explain the reason.
Yes I do see an ncreased speed up. Increasing threads decreases runtime. Each thread processes fewer rows, 
so computation is better parallelized across GPU cores. The matrix-vector multiplication is highly parallel 
and takes a lot of compute, allowing effective utilization of GPU hardware.

----------------------------------------------------------------------------


Report for Question 2 
List your code change for this question

Implemented in mult_vec_async:
Initialize y = x
Perform r = 5 asynchronous updates per launch
y[i] = d[i] + sum A[i,j] * y[j] 
Compute difference diff[i] = |x[i] − y[i]|
Host loop adds r iterations per launch and stops early 
if all diff is low enough


Let the default number of asynchronous iterations be 5 in a batch as specified in it_mult_vec.h.

List reported parallel time and the number of actual iterations executed  for n=4K, t=1K, 8x128  threads with asynchronous Gauss Seidel
Time: 0.038598 s
Actual iterations executed: 15


List reported parallel time and the number of actual iterations executed  for n=4K, t=1K,  32x128 threads with asynchronous Gauss Seidel
Time: 0.439235 s
Actual iterations executed: 1025


Is the number of iterations  executed by  above parallel asynchronous Gauss Seidel-Seidel method  bigger or smaller  than that
of the sequential Gauss Seidel-Seidel code under the same converging error threshold (1e-3)?  
Explain the reason based on the running trace of above two thread configurations that more threads may not yield more time reduction in this case. 

The parallel asynchronous Gauss-Seidel can require more iterations than sequential Gauss-Seidel.
Sequential updates rows in order and always uses the newest values which can give fast convergence.
In the parallel async version, threads update rows simultaneously and read partially only updated values. 
This weakens convergence and may increase iterations.

This basically means more threads do not always reduce runtime 
because increased parallelism increases inconsistency in updates. 
The trace shows 32×128 threads required far more iterations than 8×128 threads
so additional parallelism slowed convergence enough to outweigh the extra compute power.


Make sure you attach the  output trace  of your code below in running the tests of the unmodified it_mult_vec_test.cu on Expanse GPU for Q1 and Q2

>>>>>>>>>>>>>>>>>>>>>>>>>
Start running itmv tests.
>>>>>>>>>>>>>>>>>>>>>>>>>

Test 1:n=4, t=1, 1x2 threads:
With totally 1*2 threads, matrix size being 4, t being 1
Time cost in seconds: 0.111979
Final error (|y-x|): 1.750000.
# of iterations executed: 1.
Final y[0]=1.750000. y[n-1]=1.750000

Test 2:n=4, t=2, 1x2 threads:
With totally 1*2 threads, matrix size being 4, t being 2
Time cost in seconds: 0.000267
Final error (|y-x|): 1.312500.
# of iterations executed: 2.
Final y[0]=0.437500. y[n-1]=0.437500

Test 3:n=8, t=1, 1x2 threads:
With totally 1*2 threads, matrix size being 8, t being 1
Time cost in seconds: 0.000235
Final error (|y-x|): 1.875000.
# of iterations executed: 1.
Final y[0]=1.875000. y[n-1]=1.875000

Test 4:n=8, t=2, 1x2 threads:
With totally 1*2 threads, matrix size being 8, t being 2
Time cost in seconds: 0.000254
Final error (|y-x|): 1.640625.
# of iterations executed: 2.
Final y[0]=0.234375. y[n-1]=0.234375

Test 8a:n=4, t=1, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 4, t being 1
Time cost in seconds: 0.000244
Final error (|y-x|): 1.000193.
# of iterations executed: 5.
Final y[0]=1.000089. y[n-1]=1.000193

Test 8b:n=4, t=2, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 4, t being 2
Time cost in seconds: 0.000236
Final error (|y-x|): 1.000193.
# of iterations executed: 5.
Final y[0]=1.000089. y[n-1]=1.000193

Test 8c:n=8, t=1, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 8, t being 1
Time cost in seconds: 0.000236
Final error (|y-x|): 1.001155.
# of iterations executed: 5.
Final y[0]=1.001155. y[n-1]=0.999790

Test 8d:n=8, t=2, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 8, t being 2
Time cost in seconds: 0.000243
Final error (|y-x|): 1.001155.
# of iterations executed: 5.
Final y[0]=1.001155. y[n-1]=0.999790

Test 9: n=4K t=1K 32x128 threads:
With totally 32*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.560899
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225

Test 9a: n=4K t=1K 16x128 threads:
With totally 16*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 1.033989
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225

Test 9b: n=4K t=1K 8x128 threads:
With totally 8*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 2.047910
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225

Test 9c: n=4K t=1K 4x128 threads:
With totally 4*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 4.102072
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225

Test 11: n=4K t=1K 32x128 threads/Async:
With totally 32*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.439235
Final error (|y-x|): 0.002786.
# of iterations executed: 1025.
Final y[0]=1.001344. y[n-1]=1.001373

Test 11a: n=4K t=1K 8x128 threads/Async:
With totally 8*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.038598
Final error (|y-x|): 0.000000.
# of iterations executed: 15.
Early exit due to convergence, even asked for 1024 iterations.
Asynchronous code actually runs 15 iterations.
Final y[0]=1.000000. y[n-1]=1.000000

Summary: Failed 0 out of 14 tests

----------------------------------------------------------------------------

Report for Question 3

List your solution to call  cublasDgemm() in Method 1.

cublasDgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_A, N,
            d_B, N,
            &beta,
            d_C, N);

List the latency and GFLOPs of the above 3 version of implementation and the number of Cuda threads used in executing Method 3 
when matrix dimension N varies as 50, 200, 800,  and 1600.  

N = 50
cuBLAS DGEMM: 0.024 ms, 10.615 GFLOPs
cuBLAS DGEMV loop: 0.157 ms, 1.596 GFLOPs
Naive kernel: 0.009 ms, 27.127 GFLOPs
Method 3 threads: 20×20 = 400 threads per block, grid 3×3 → 3,600 threads

N = 200
cuBLAS DGEMM: 0.041 ms, 390.625 GFLOPs
cuBLAS DGEMV loop: 3.406 ms, 4.698 GFLOPs
Naive kernel: 0.029 ms, 558.036 GFLOPs
Method 3 threads: grid 10×10 → 40,000 threads

N = 800
cuBLAS DGEMM: 0.236 ms, 4347.826 GFLOPs
cuBLAS DGEMV loop: 13.580 ms, 75.403 GFLOPs
Naive kernel: 1.091 ms, 938.967 GFLOPs
Method 3 threads: grid 40×40 → 640,000 threads

N = 1600
cuBLAS DGEMM: 1.303 ms, 6289.308 GFLOPs
cuBLAS DGEMV loop: 46.523 ms, 176.083 GFLOPs
Naive kernel: 7.628 ms, 1073.970 GFLOPs
Method 3 threads: grid 80×80 → 2,560,000 threads



List the highest gigaflops you have observed with V100 from this question and the highest gigaflops  you have observed from PA2 MKL GEMM code  when N=1600.  
Compute the ratio between these two numbers as the speedup of V100 over a CPU host. 

Highest V100 performance observed:
6289.308 GFLOPs (cuBLAS DGEMM at N = 1600)

Highest PA2 MKL GEMM performance at N = 1600:
152.29 GFLOPs

Speedup of GPU over CPU:
Speedup = 6289.308 / 152.29 which is about  41.29 times
