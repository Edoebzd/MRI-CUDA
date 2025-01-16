#include "utils.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>

using namespace std;

int next_power_of_two(int N) {
    if (N <= 1) return 1; // Edge case for numbers <= 1
    int power = 1;
    while (power < N) {
        power <<= 1; // Equivalent to power = power * 2;
    }
    return power;
}

void write_to_png(float* data, unsigned int size, string outfile) {

    char* filename = const_cast<char*>(outfile.c_str());
    unsigned int rows = size;
    unsigned int cols = size;

    // Crea un array di byte per l'immagine
   unsigned char imageData[rows * cols];

    // trova la massima magnitudine per normalizzare i valori
    double maxMagnitude = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (data[i*size + j] > maxMagnitude) {
                maxMagnitude = data[i*size + j];
            }
        }
    }

    // Normalizza le magnitudini e converte in byte
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            unsigned char pixelValue = static_cast<unsigned char>((data[i*size + j] / maxMagnitude) * 255);
            imageData[i * cols + j] = pixelValue;
        }
    }


    // Scrive il risultato come PNG
    if (stbi_write_png(filename, cols, rows, 1, imageData, cols) == 0) {
        cout << "Errore nella scrittura dell'immagine in " << filename << endl;
        throw runtime_error("Errore nella scrittura dell'immagine");
    }

    cout << "Immagine salvata come " << filename << endl;
}