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
#include "thrust/complex.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <thread>
#define PI 3.14159265358979323846

using namespace std;

#define FFT_SH_MEM_PADDING 1 // Shared memory padding to decrease bank conflicts
#define WARP_SIZE 32

#define FULL_MASK 0xFFFFFFFF
#define MASK_0_TO_15 0x0000FFFF

#define index(slice, ch, row, col, size, n_ch) ((n_ch * size * size * slice) + (size * size * ch) + (size * row) + col)
#define sliceIndex (slice * num_channels * size * size)

__constant__ int numChannels_gpu;
__constant__ int size_gpu;
__constant__ int size2_gpu;
__constant__ int sizeLog2_gpu;
__constant__ int sizeSq_gpu;

void writePNG(string outDir, int slice, unsigned char *imgData, int size) {
    if (stbi_write_png((outDir + to_string(slice) + ".png").c_str(), size, size, 1, imgData, size) == 0) {
        cerr << "Errore nella scrittura dell'immagine in " << outDir + to_string(slice) + ".png" << endl;
        throw runtime_error("Errore nella scrittura dell'immagine");
    }
    cout << "Immagine salvata come " << outDir + to_string(slice) + ".png" << endl;
}

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double) ts.tv_sec + (double) ts.tv_nsec * 1.e-9);
}

__device__ uint32_t reverse_bits_gpu(uint32_t x) {
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

#define padded(x) ((x) + ((x)/WARP_SIZE)*FFT_SH_MEM_PADDING)
#define threadId threadIdx.x
#define rowIndex (blockIdx.x * size_gpu)
#define channelIndex (blockIdx.y * sizeSq_gpu)
__global__ void kernel_fft(thrust::complex<float> *data) {
    extern __shared__ thrust::complex<float> data_shared[];
    int baseIndex = channelIndex + rowIndex + threadId;

    data_shared[padded(reverse_bits_gpu(threadId) >> (32 - sizeLog2_gpu))] =
            data[baseIndex];
    data_shared[padded(reverse_bits_gpu(threadId+size2_gpu) >> (32 - sizeLog2_gpu))] =
            data[baseIndex + size2_gpu];

    __syncthreads();

    int k, j;
    float twiddle_real, twiddle_img;
    thrust::complex<float> t;
    for (int level = 1, step = 1; level <= sizeLog2_gpu; level++, step *= 2) {
        j = threadId % step;
        k = (threadId / step * (1 << level)) + j;


        sincosf(-(float) PI * j / step, &twiddle_img, &twiddle_real);
        t = thrust::complex<float>(twiddle_real, twiddle_img) * data_shared[padded(k+step)];
        data_shared[padded(k+step)] = data_shared[padded(k)] - t;
        data_shared[padded(k)] = data_shared[padded(k)] + t;

        __syncthreads();
    }
    data[baseIndex] = data_shared[padded(threadId)];
    data[baseIndex + size2_gpu] = data_shared[padded(threadId+size2_gpu)];
}
#undef threadId
#undef rowIndex
#undef channelIndex


#define id threadIdx.x
#define miniBlockCol threadIdx.y
#define miniBlockRow threadIdx.z
#define blockCol blockIdx.x
#define blockRow blockIdx.y
#define channelIndex (blockIdx.z * sizeSq_gpu)
#define idMod4 (id & 3) //id % 4
#define idDiv4 (id >> 2) //id / 4
__global__ void kernel_transpose(thrust::complex<float> *data) {
    //indici degli scambi effettuati con shuffle, precalcolati
    const int lookup_table[32] = {
        16, 20, 24, 28, 17, 21, 25, 29,
        18, 22, 26, 30, 19, 23, 27, 31,
        0, 4, 8, 12, 1, 5, 9, 13,
        2, 6, 10, 14, 3, 7, 11, 15
    };

    int index;
    thrust::complex<float> tmp;
    if (blockRow > blockCol) return;
    if (blockRow == blockCol) {
        if (miniBlockRow > miniBlockCol) return;
        if (miniBlockRow == miniBlockCol) {
            if (id >= 16) return;
            index = channelIndex
                    + (blockRow * 16 + miniBlockRow * 4 + idDiv4) * size_gpu
                    + (blockCol * 16 + miniBlockCol * 4 + idMod4);

            tmp = data[index];
            tmp.real(__shfl_sync(MASK_0_TO_15, tmp.real(), lookup_table[id + 16], 16));
            tmp.imag(__shfl_sync(MASK_0_TO_15, tmp.imag(), lookup_table[id + 16], 16));
            data[index] = tmp;
            return;
        }
    }

    if (id < 16) {
        index = channelIndex
                + (blockRow * 16 + miniBlockRow * 4 + idDiv4) * size_gpu
                + (blockCol * 16 + miniBlockCol * 4 + idMod4);
    } else {
        index = channelIndex
                + (blockCol * 16 + miniBlockCol * 4 + idDiv4 - 4) * size_gpu
                + (blockRow * 16 + miniBlockRow * 4 + idMod4);
    }

    tmp = data[index];
    tmp.real(__shfl_sync(FULL_MASK, tmp.real(), lookup_table[id], 32));
    tmp.imag(__shfl_sync(FULL_MASK, tmp.imag(), lookup_table[id], 32));
    data[index] = tmp;
}
#undef id
#undef miniBlockCol
#undef miniBlockRow
#undef blockCol
#undef blockRow
#undef channelIndex_gpu
#undef idMod4
#undef idDiv4

#define colId threadIdx.x
#define rowId blockIdx.x
__global__ void kernel_combineChannels(thrust::complex<float> *data, float *max) {
    extern __shared__ float shared_data[];

    //sum of channels
    int baseIndex = rowId * size_gpu + colId;
    float sum = 0, tmp;
    for (int ch = 0; ch < numChannels_gpu; ch++) {
        tmp = thrust::abs(data[baseIndex + ch * sizeSq_gpu]);
        sum += tmp * tmp;
    }
    shared_data[colId] = sqrtf(sum);
    data[baseIndex].real(shared_data[colId]);

    __syncthreads();

    //find max for each row
    for (int stride = size2_gpu; stride > 0; stride /= 2) {
        if (colId >= stride) return;
        tmp = shared_data[colId + stride];
        if (shared_data[colId] < tmp) {
            shared_data[colId] = tmp;
        }
        __syncthreads();
    }
    max[rowId] = shared_data[0];
}

__global__ void kernel_max(float *data) {
    //find max between the max of each row
    extern __shared__ float shared_data[];

    shared_data[threadIdx.x] = data[threadIdx.x];
    shared_data[threadIdx.x + size2_gpu] = data[threadIdx.x + size2_gpu];

    int stride = size2_gpu;
    float tmp;
    tmp = shared_data[threadIdx.x + stride];
    if (shared_data[threadIdx.x] < tmp) {
        shared_data[threadIdx.x] = tmp;
    }
    __syncthreads();
    for (stride /= 2; stride > 0; stride /= 2) {
        if (threadIdx.x >= stride) return;
        tmp = shared_data[threadIdx.x + stride];
        if (shared_data[threadIdx.x] < tmp) {
            shared_data[threadIdx.x] = tmp;
        }
        __syncthreads();
    }
    data[0] = shared_data[0];
}

__global__ void kernel_shiftToChar(const thrust::complex<float> *__restrict__ data, const float *__restrict__ max, unsigned char *imgData) {
    imgData[(size_gpu - 1 - ((rowId + size2_gpu) % size_gpu)) * size_gpu + (size_gpu - 1 - ((colId + size2_gpu) % size_gpu))] =
            (unsigned char) ((data[rowId * size_gpu + colId].real() / (*max)) * 255);
}
#undef colId
#undef rowId


int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input_file> <output_dir>" << endl;
        return 1;
    }

    cout << "Lettura del file..." << endl;

    string datafile = argv[1];
    ISMRMRD::Dataset d(datafile.c_str(), "dataset", false);

    unsigned int num_acquisitions = d.getNumberOfAcquisitions();
    cout << "Number of acquisitions: " << num_acquisitions << endl;

    ISMRMRD::Acquisition acq;
    d.readAcquisition(0, acq);
    int num_channels = acq.active_channels();
    int num_samples = acq.number_of_samples();
    int num_slices = num_acquisitions / num_samples;

    cout << "Number of channels: " << num_channels << endl;
    cout << "Number of samples: " << num_samples << endl;
    cout << "Number of slices: " << num_slices << endl;

    int size = next_power_of_two(num_samples);

    cout << "Loading data..." << endl;

    //host malloc + memset a 0
    thrust::complex<float> *data;
    cudaMallocHost((void **) &data, num_slices * num_channels * size * size * sizeof(thrust::complex<float>));
    memset(data, 0, size * size * num_slices * num_channels * sizeof(thrust::complex<float>));

    //reading all the data with padding
    int pad = (size - num_samples) / 2;

    for (int slice = 0; slice < num_slices; slice++) {
        for (int row = 0; row < num_samples; row++) {
            d.readAcquisition(slice * num_samples + row, acq);
            for (int channel = 0; channel < num_channels; channel++) {
                for (int col = 0; col < num_samples; col++) {
                    data[index(slice, channel, (row+pad), (col+pad), size, num_channels)] = (thrust::complex<float>) (
                        acq.data(col, channel));
                }
            }
        }
    }

    cout << "ok" << endl;

    int constant_tmp;
    constant_tmp = num_channels;
    cudaMemcpyToSymbol(numChannels_gpu, &constant_tmp, sizeof(int));
    constant_tmp = size;
    cudaMemcpyToSymbol(size_gpu, &constant_tmp, sizeof(int));
    constant_tmp = size / 2;
    cudaMemcpyToSymbol(size2_gpu, &constant_tmp, sizeof(int));
    constant_tmp = log2(size);
    cudaMemcpyToSymbol(sizeLog2_gpu, &constant_tmp, sizeof(int));
    constant_tmp = size * size;
    cudaMemcpyToSymbol(sizeSq_gpu, &constant_tmp, sizeof(int));

    thrust::complex<float> *data_gpu;
    cudaMalloc((void **) &data_gpu, num_slices * num_channels * size * size * sizeof(thrust::complex<float>));
    float *tmpMax_gpu;
    cudaMalloc((void **) &tmpMax_gpu, num_slices * size * sizeof(float));
    unsigned char *imgData_gpu;
    cudaMalloc((void **) &imgData_gpu, num_slices * size * size * sizeof(unsigned char));
    unsigned char *imgData;
    cudaMallocHost((void **) &imgData, num_slices * size * size * sizeof(unsigned char));

    cout << "ok" << endl;

    cudaStream_t stream[num_slices];


    dim3 grid_fft(size, num_channels);
    dim3 block_fft(size / 2);

    dim3 grid_transpose(size / 4 / 4, size / 4 / 4, num_channels);
    dim3 block_transpose(WARP_SIZE, 4, 4);

    double iStart = cpuSecond();

    for (int slice = 0; slice < num_slices; slice++) {
        cudaStreamCreate(&stream[slice]);

        cudaMemcpyAsync(data_gpu + sliceIndex, data + sliceIndex, num_channels * size * size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice, stream[slice]);

        kernel_fft <<<grid_fft, block_fft, (size + (size / WARP_SIZE) * FFT_SH_MEM_PADDING) * sizeof(thrust::complex<float>), stream[slice]>>>(data_gpu + sliceIndex);
        kernel_transpose <<<grid_transpose, block_transpose,0, stream[slice]>>>(data_gpu + sliceIndex);
        kernel_fft <<<grid_fft, block_fft, (size + (size / WARP_SIZE) * FFT_SH_MEM_PADDING) * sizeof(thrust::complex<float>), stream[slice]>>>(data_gpu + sliceIndex);

        kernel_combineChannels <<<size, size, size * sizeof(float), stream[slice]>>>(data_gpu + sliceIndex, tmpMax_gpu + slice * size);
        kernel_max <<<1, size / 2, size * sizeof(float), stream[slice]>>>(tmpMax_gpu + slice * size);
        kernel_shiftToChar <<<size, size, 0, stream[slice]>>>(data_gpu + sliceIndex, tmpMax_gpu + slice * size, imgData_gpu + slice * size * size);

        cudaMemcpyAsync(imgData + slice * size * size, imgData_gpu + slice * size * size, size * size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream[slice]);

        cudaStreamDestroy(stream[slice]);
    }

    cudaDeviceSynchronize();

    double iElaps = cpuSecond() - iStart;
    cout << "Elapsed time: " << iElaps << " s" << endl;


    //write img to disk (parallel with C++ threads
    vector<thread> threads;
    for (int slice = 0; slice < num_slices; slice++) {
        threads.emplace_back(writePNG, argv[2], slice, imgData + slice * size * size, size);
    }
    for (int slice = 0; slice < num_slices; slice++) {
        threads[slice].join();
    }

    cudaFree(tmpMax_gpu);
    cudaFree(imgData_gpu);
    cudaFreeHost(imgData);
    cudaFreeHost(data);
    cudaFree(data_gpu);

    return 0;
}
