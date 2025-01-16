#include <iostream>
#include <fstream>
#include <complex>
#include <string>
#include <cmath>
#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"
#include "ismrmrd/xml.h"
#include "fft/fft-cuda.cuh"
#include "utils.h"

using namespace std;

int main(int argc, char** argv) {

	if(argc != 2) {
		cout << "Incorrect invocation, argument must be <dataFile>" << endl;
		return -1;
	}
    cout << "Lettura del file "<< argv[1] <<" ... " << endl;

    string datafile = argv[1];

    ISMRMRD::Dataset d(datafile.c_str(), "dataset", false);
    string xml;
    d.readHeader(xml);
    ISMRMRD::IsmrmrdHeader hdr;
    //ISMRMRD::deserialize(xml.c_str(), hdr);

    unsigned int num_acquisitions = d.getNumberOfAcquisitions();
    cout << "Number of acquisitions: " << num_acquisitions << endl;

    ISMRMRD::Acquisition acq;
    d.readAcquisition(0, acq);
    unsigned int num_channels = acq.active_channels();
    unsigned int num_samples = acq.number_of_samples();
    unsigned int num_slices = num_acquisitions / num_samples;

    // width and height of the slice
    unsigned int width = num_samples;
    unsigned int height = num_samples;

    cout << "Number of channels: " << num_channels << endl;
    cout << "Number of samples: " << num_samples << endl;
    cout << "Number of slices: " << num_slices << endl;
	cout << "width: " << width << endl;
	cout << "height: " << height << endl;

    // 3D array to store the multi channel slice data
    // num_channels x width x height

	unsigned int padded_width = next_power_of_two(width);
	unsigned int padded_height = next_power_of_two(height);
	if (padded_height != padded_width) {
		cout << "Error: the width and height must be equal" << endl;
		return 1;
	}
	unsigned int n = padded_width;
	unsigned int padding = (n - width) / 2;
	unsigned int vpadding = padding*n;

	float dataR[num_channels][n*n];
	float dataI[num_channels][n*n];
	memset(dataR, 0, num_channels * n * n * sizeof(float));
	memset(dataI, 0, num_channels * n * n * sizeof(float));

	//load data
	complex_float_t tmp;
	for (unsigned int slice = 0; slice < num_slices; slice++) {

		for(unsigned int j = 0; j < num_samples; j++) {
			d.readAcquisition(slice*num_samples + j, acq);
			for (unsigned int channel = 0; channel < num_channels; channel++) {
				for (unsigned int sample = 0; sample < num_samples; sample++) {
					tmp = acq.data(sample, channel);
					dataR[channel][vpadding + j*n + padding + sample] = tmp.real();
					dataI[channel][vpadding + j*n + padding + sample] = tmp.imag();
				}
			}
		}

		cout << "Processing data..." << endl;

        // ---------------------------------- OPTIONAL --------------------------------------------
		// comment to speed up the process
		// write the k-space data to a PNG file (only the magnitude of the first channel)

  //      string kspaceFile = "C:/Users/user/source/repos/FFT/output/kspace/" + to_string(slice) + ".png";
  //      vector<vector<double>> k_space_image(padded_width, vector<double>(padded_height, 0.0));
		//combineCoils(slice_channels_padded, k_space_image, padded_width, padded_height, num_channels);
		//apply_scale(k_space_image);
  //      write_to_png(k_space_image, kspaceFile);

        // -----------------------------------------------------------------------------------------


        // 2D IFFT
		for (unsigned int channel = 0; channel < num_channels; channel++) {

			FFT2D_GPU(dataR[channel], dataI[channel], n, 1.0);
			//FFT_SHIFT(slice_channels_padded[channel], padded_width, padded_height);
		}

		// final vector to store the image
		float image[n*n];

		// combine the channels
		float sumOfSquares, magnitude;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {

				sumOfSquares = 0.0;
				for (int k = 0; k < num_channels; k++) {
					magnitude = sqrt(dataR[k][i*n + j] * dataR[k][i*n + j] + dataI[k][i*n + j] * dataI[k][i*n + j]);
					sumOfSquares += magnitude * magnitude;
				}
				image[i*n +j] = sqrt(sumOfSquares);
			}
		}




		// rotate the image by 90 degrees
		//rotate_90_degrees(mri_image);

		// flip
		//flipVertical(mri_image, padded_width, padded_height);
		//flipHorizontal(mri_image, padded_width, padded_height);

		string magnitudeFile = "../output/images/" + to_string(slice) + ".png";

		write_to_png(image, n, magnitudeFile);

	} // end for slice


    return 0;

}
