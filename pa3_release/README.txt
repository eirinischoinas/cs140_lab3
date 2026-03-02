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

----------------------------------------------------------------------------

Report for Question 3

List your solution to call  cublasDgemm() in Method 1.

List the latency and GFLOPs of the above 3 version of implementation and the number of Cuda threads used in executing Method 3 
when matrix dimension N varies as 50, 200, 800,  and 1600.  


List the highest gigaflops you have observed with V100 from this question and the highest gigaflops  you have observed from PA2 MKL GEMM code  when N=1600.  
Compute the ratio between these two numbers as the speedup of V100 over a CPU host. 

