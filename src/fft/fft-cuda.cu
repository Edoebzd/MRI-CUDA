#include "fft-cuda.cuh"
#include <thrust/complex.h>
#include <complex>
#include <math.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define FORWARD 1
#define REVERSE -1

bool MYpowerOf2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

__device__ void FFT1D(short dir, thrust::complex<float>* data, int length) {

    long n = 1 << length; // 2^length, numero di punti della FFT

    //applica il bit reversal al vettore
    int i2 = n >> 1; // n/2
    int k, j = 0;
    thrust::complex<float> tmp;
    for (int i=0; i<n-1; i++) {
        if (i < j) {
            tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
        k = i2;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j+=k;
    }

    //FFT (algoritmo cooley-tukey)
    int i, i1, l, l1, l2 = 1;
    thrust::complex<float> u, t;
    thrust::complex<float> c = {-1.0, 0.0};
    for (l=0;l<length;l++) {
        l1 = l2;
        l2 <<= 1;
		u = thrust::complex<float>(1.0, 0.0);
        for (j=0;j<l1;j++) {
            for (i=j;i<n;i+=l2) {
                i1 = i + l1;
                t = u * data[i1];
                data[i1] = data[i] - t;
                data[i] += t;
            }
            u = u * c;
        }

        c.imag(sqrt((1.0 - c.real()) / 2.0));
        if (dir == 1)
            c.imag(-c.imag());
        c.real(sqrt((1.0 + c.real()) / 2.0));
    }

    /* Scaling for forward transform */
    if (dir == 1) {
        for (i=0;i<n;i++) {
            data[i].real(data[i].real()/n);
            data[i].imag(data[i].imag()/n);
        }
    }

}

__global__ void FFT2D_GPU_RIGHE(thrust::complex<float>* data, thrust::complex<float>* temp, int n, int nlog2, short dir) {
    FFT1D(dir, data +(threadIdx.x*n), nlog2);
    for(int i = 0; i < n; i++) {
        temp[i*n + threadIdx.x] = data[threadIdx.x*n + i];
    }
}

__global__ void FFT2D_GPU_COLONNE(thrust::complex<float>* data, thrust::complex<float>* temp, int n, int nlog2, short dir) {
    FFT1D(dir, temp + (threadIdx.x*n), nlog2);
    for(int i = 0; i < n; i++) {
        data[i*n + threadIdx.x] = temp[threadIdx.x*n + i];
    }
  }

__global__ void FFT2D_GPU_COMPUTE(thrust::complex<float>* data, thrust::complex<float>* temp, int n, int nlog2, short dir) {

	// FFT delle righe
    FFT1D(dir, data + (threadIdx.x * n), nlog2);
    for (int i = 0; i < n; i++) {
        temp[i * n + threadIdx.x] = data[threadIdx.x * n + i];
    }

    __syncthreads();

	// FFT delle colonne
    FFT1D(dir, temp + (threadIdx.x * n), nlog2);
    for (int i = 0; i < n; i++) {
        data[i * n + threadIdx.x] = temp[threadIdx.x * n + i];
    }
}

__global__ void FFT_SHIFT_GPU(thrust::complex<float>* data, thrust::complex<float>* temp, int n) {

	int n2 = n / 2;
    // Shift delle righe
    for (int i = 0; i < n2; i++) {
		temp[threadIdx.x*n + i + n2] = data[threadIdx.x * n + i];
		temp[threadIdx.x * n + i] = data[threadIdx.x * n + i + n2];
    }

	// Barriera per sincronizzare i thread
    __syncthreads();

    // Shift delle colonne
    for (int i = 0; i < n2; i++) {
		data[i * n + threadIdx.x] = temp[(i + n2) * n + threadIdx.x];
		data[(i + n2) * n + threadIdx.x] = temp[i * n + threadIdx.x];
    }
}

bool FFT2D_GPU(thrust::complex<float>* data, int n, short dir) {

    int nlog2 = log2(n);


    thrust::complex<float>* data_gpu;
	thrust::complex<float>* temp_gpu;
    cudaMalloc((void**)&data_gpu, n * n * sizeof(thrust::complex<float>));
    cudaMalloc((void**)&temp_gpu, n * n * sizeof(thrust::complex<float>));
    cudaMemcpy(data_gpu, data, n * n * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(n);

    FFT2D_GPU_COMPUTE <<<grid, block >>> (data_gpu, temp_gpu, n, nlog2, dir);
    cudaDeviceSynchronize();

    //FFT SHIFT
	FFT_SHIFT_GPU << <grid, block >> > (data_gpu, temp_gpu, n);
	cudaDeviceSynchronize();

    cudaMemcpy(data, data_gpu, n * n * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
    cudaFree(data_gpu);
    cudaFree(temp_gpu);

    // converto nuovamente i dati
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			data[i*n+j] = std::complex<float>(data[i * n + j].real(), data[i * n + j].imag());
		}
	}

    return true;
}


void FFT_SHIFT(std::vector<std::vector<std::complex<float>>>& array, int rows, int cols) {
    // Shift delle righe
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols / 2; ++j) {
            std::swap(array[i][j], array[i][j + cols / 2]);
        }
    }
    // Shift delle colonne
    for (int i = 0; i < rows / 2; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::swap(array[i][j], array[i + rows / 2][j]);
        }
    }
}
