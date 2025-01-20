
#include <complex>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/complex.h>

bool FFT2D_GPU(thrust::complex<float>* data, int n, short dir);
__device__ void FFT1D(short dir, thrust::complex<float>* data, int length);