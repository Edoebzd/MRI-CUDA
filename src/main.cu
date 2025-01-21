#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <chrono>
#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"
#include "ismrmrd/serialization.h"
#include "ismrmrd/xml.h"
#include "utils.h"
#include "fft-cuda.cuh"
#include "thrust/complex.h"

#define index(slice, ch, row, col, size, n_ch) ((n_ch * size * size * slice) + (size * size * ch) + (size * row) + col)

//#define index(slice, ch, row, col, size, n_ch) 1

using namespace std;



__global__ void kernel(thrust::complex<float>* data, int size, int sizelog2, int dir, thrust::complex<float>* temp) {
    //extern __shared__ thrust::complex<float> temp[];

    //row fft
    FFT1D(dir, data + ((blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + (threadIdx.x * size)), sizelog2);
    for(int i = 0; i < size; i++) {
        temp[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + i * size + threadIdx.x] = data[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + threadIdx.x * size + i];
    }
    __syncthreads();

    //col fft
    FFT1D(dir, temp + (blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + (threadIdx.x * size), sizelog2);
    for(int i = 0; i < size; i++) {
        data[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + (i * size + threadIdx.x)] = temp[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + threadIdx.x * size + i];
    }
    __syncthreads();

    //shift
    int size2 = size / 2;
    //row shift
    for (int i = 0; i < size2; i++) {
        temp[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + threadIdx.x*size + i + size2] = data[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + threadIdx.x * size + i];
        temp[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + threadIdx.x * size + i] = data[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + threadIdx.x * size + i + size2];
    }
    __syncthreads();

    //col shift
    for (int i = 0; i < size2; i++) {
        data[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + i * size + threadIdx.x] = temp[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + (i + size2) * size + threadIdx.x];
        data[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + (i + size2) * size + threadIdx.x] = temp[(blockIdx.y * gridDim.x * size * size) + (blockIdx.x * size * size) + i * size + threadIdx.x];
    }
}

int main(int argc, char* argv[]) {

    cout << "Lettura del file..." << endl;

    string datafile = argv[1];

    ISMRMRD::Dataset d(datafile.c_str(), "dataset", false);

    unsigned int num_acquisitions = d.getNumberOfAcquisitions();
    cout << "Number of acquisitions: " << num_acquisitions << endl;

    ISMRMRD::Acquisition acq;
    d.readAcquisition(0, acq);
    unsigned int num_channels = acq.active_channels();
    unsigned int num_samples = acq.number_of_samples();
    unsigned int num_slices = num_acquisitions / num_samples;

    // width and height of the slice

    cout << "Number of channels: " << num_channels << endl;
    cout << "Number of samples: " << num_samples << endl;
    cout << "Number of slices: " << num_slices << endl;

    // padded array to perform FFT
    unsigned int size = next_power_of_two(num_samples);
    int sizelog2 = log2(size);

    cout << "Loading data..." << endl;
    // Read the data from the acquisitions

    thrust::complex<float>* data;

	data = (thrust::complex<float>*)malloc(size * size * num_slices * num_channels * sizeof(thrust::complex<float>));

	memset(data, 0, size * size * num_slices * num_channels * sizeof(thrust::complex<float>));

    //reading all the data with padding

	complex<float> tmp = complex<float>(0.0, 0.0);
	int pad = (size - num_samples) / 2;

    for (int slice = 0; slice < num_slices; slice++) {
        for (int row = 0; row < num_samples; row++) {
			d.readAcquisition(slice * num_samples + row, acq);
            for (int channel = 0; channel < num_channels; channel++) {
                for (int col = 0; col < num_samples; col++) {
                    tmp = acq.data(col, channel);
					data[index(slice, channel, row+pad, col+pad, size, num_channels)] = thrust::complex<float>(tmp.real(), tmp.imag());
                }
            }
        }

    }

    //load to gpu
    thrust::complex<float>* data_gpu;
    thrust::complex<float>* temp_gpu;
    cudaMalloc((void**)&data_gpu, num_slices * num_channels * size * size * sizeof(thrust::complex<float>));
    cudaMalloc((void**)&temp_gpu, num_slices * num_channels * size * size * sizeof(thrust::complex<float>));
    cudaMemcpy(data_gpu, data, num_slices * num_channels * size * size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);

    //gpu grid
    dim3 grid(num_channels, num_slices);
    dim3 block(size);

    cout << "Computing ..." << endl;
    auto start = std::chrono::high_resolution_clock::now();

    kernel<<<grid, block>>>(data_gpu, size, sizelog2, 1, temp_gpu);
    cudaDeviceSynchronize();

    cudaMemcpy(data, data_gpu, num_slices * num_channels * size * size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);
    cudaFree(data_gpu);
    cudaFree(temp_gpu);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tempo impiegato: " << duration_ms.count() << " millisecondi" << std::endl;
    cout << "Saving images..." << endl;

    for (int slice = 0; slice < num_slices; slice++) {
        /*
        // 2D IFFT
        auto start = std::chrono::high_resolution_clock::now();
        for (int channel = 0; channel < num_channels; channel++) {


			FFT2D_GPU( data + index(slice, channel, 0, 0, size, num_channels), 512, 1);

            //FFT_SHIFT(slice_channels_padded[channel], padded_width, padded_height);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Tempo impiegato: " << duration_ms.count() << " millisecondi" << std::endl;
        */

        // final vector to store the image
        vector<vector<float>> mri_image(size, vector<float>(size, 0.0));

        // combine the coils
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                float sumSquares = 0.0;
                for (int ch = 0; ch < num_channels; ++ch) {
                    // Magnitudine del valore complesso per il coil k
                    float magnitude = abs(data[index(slice,ch,row,col,size,num_channels)]);
                    sumSquares += magnitude * magnitude;
                }
                // Calcola il risultato RSS
                mri_image[row][col] = sqrt(sumSquares);
            }
        }


        // rotate the image by 90 degrees
        //rotate_90_degrees(mri_image);

        // flip
        //flipVertical(mri_image, padded_width, padded_height);
        //flipHorizontal(mri_image, padded_width, padded_height);

        string magnitudeFile = argv[2] + to_string(slice) + ".png";

        write_to_png(mri_image, magnitudeFile);
    } // end for slice


    return 0;

}