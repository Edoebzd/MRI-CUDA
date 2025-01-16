#include "fft-cuda.cuh"

#include <math.h>

using namespace std;

__device__ void FFT1D(short dir, float* dataR, float* dataI, int length) {

    long n = 1 << length; // 2^length, numero di punti della FFT

    //applica il bit reversal al vettore
    int i2 = n >> 1; // n/2
    int k, j = 0;
    float tR, tI;
    for (int i=0; i<n-1; i++) {
        if (i < j) {
            tR = dataR[i];
            tI = dataI[i];
            dataR[i] = dataR[j];
            dataI[i] = dataI[j];
            dataR[j] = tR;
            dataI[j] = tI;
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
    float uR, uI;
    float cR = -1.0, cI = 0.0;
    for (l=0;l<length;l++) {
        l1 = l2;
        l2 <<= 1;
        uR = 1.0;
        uI = 0.0;
        for (j=0;j<l1;j++) {
            for (i=j;i<n;i+=l2) {
                i1 = i + l1;
                tR = uR * dataR[i1] - uI * dataI[i1];
                tI = uR * dataI[i1] + uI * dataR[i1];
                dataR[i1] = dataR[i] - tR;
                dataI[i1] = dataI[i] - tI;
                dataR[i] += tR;
                dataI[i] += tI;
            }
            tR = uR * cR - uI * cI;
            uI = uR * cI + uI * cR;
            uR = tR;
        }

        cI = sqrt((1.0 - cR) / 2.0);
        if (dir == 1)
            cI = -cI;
        cR = sqrt((1.0 + cR) / 2.0);
    }

    /* Scaling for forward transform */
    if (dir == 1) {
        for (i=0;i<n;i++) {
            dataR[i] /= n;
            dataI[i] /= n;
        }
    }

}

__global__ void FFT2D_GPU_RIGHE(float* dataR, float* dataI, float* tempR, float* tempI, int n, int nlog2, short dir) {
    FFT1D(dir, dataR + (threadIdx.x*n), dataI + (threadIdx.x*n), nlog2);
    for(int i = 0; i < n; i++) {
        tempR[i*n + threadIdx.x] = dataR[threadIdx.x*n + i];
        tempI[i*n + threadIdx.x] = dataI[threadIdx.x*n + i];
    }
}

__global__ void FFT2D_GPU_COLONNE(float* dataR, float* dataI, float* tempR, float* tempI, int n, int nlog2, short dir) {
    FFT1D(dir, tempR + (threadIdx.x*n), tempI + (threadIdx.x*n), nlog2);
    for(int i = 0; i < n; i++) {
        dataR[i*n + threadIdx.x] = tempR[threadIdx.x*n + i];
        dataI[i*n + threadIdx.x] = tempI[threadIdx.x*n + i];
    }
}

bool FFT2D_GPU(float* dataR, float* dataI, int n, short dir) {

    int nlog2 = log2(n);

    float* dataR_gpu;
    float* dataI_gpu;
    float* tempR_gpu;
    float* tempI_gpu;
    cudaMalloc((void**) &dataR_gpu, n * n * sizeof(float));
    cudaMalloc((void**) &dataI_gpu, n * n * sizeof(float));
    cudaMalloc((void**) &tempR_gpu, n * n * sizeof(float));
    cudaMalloc((void**) &tempI_gpu, n * n * sizeof(float));
    cudaMemcpy(dataR_gpu, dataR, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dataI_gpu, dataI, n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(n);

    FFT2D_GPU_RIGHE<<<grid, block>>>(dataR_gpu, dataI_gpu, tempR_gpu, tempI_gpu, n, nlog2, dir);
    cudaDeviceSynchronize();

    FFT2D_GPU_COLONNE<<<grid, block>>>(dataR_gpu, dataI_gpu, tempR_gpu, tempI_gpu, n, nlog2, dir);
    cudaDeviceSynchronize();

    cudaMemcpy(dataR, dataR_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dataI, dataI_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dataR_gpu);
    cudaFree(dataI_gpu);
    cudaFree(tempR_gpu);
    cudaFree(tempI_gpu);
    return true;
}



/*
bool _FFT2D(complex<double>** data, int nRighe, int nColonne, short dir) {
    int i,j;
    // Transform the rows
    if (nRighe != nColonne || !MYpowerOf2(nRighe) || !MYpowerOf2(nColonne)) return(false);
    int n = log2(nRighe);

    complex<double>* temp1 = static_cast<complex<double>* >(malloc(nRighe * sizeof (complex<double>)));
    for (j=0;j<nColonne;j++) {
        for (i=0;i<nRighe;i++) {
            temp1[i] = data[i][j];
        }
        FFT1D(dir,temp1, n);
        for (i=0;i<nColonne;i++) {
            data[i][j] = temp1[i];
        }
    }
    free(temp1);

    // Transform the columns
    complex<double>* temp2 = static_cast<complex<double>* >(malloc(nColonne * sizeof (complex<double>)));
    for (i=0;i<nRighe;i++) {
        for (j=0;j<nColonne;j++) {
            temp2[j] = data[i][j];
        }
        FFT1D(dir,temp2, n);
        for (j=0;j<nRighe;j++) {
            data[i][j] = temp2[j];
        }
    }
    free(temp2);
    return(true);
}
*/



