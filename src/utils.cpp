#include "utils.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


void write_to_png(vector<vector<float>> data, string outfile) {

    char* filename = const_cast<char*>(outfile.c_str());
    int rows = data.size();
    int cols = data[0].size();

    // Crea un array di byte per l'immagine
    vector<unsigned char> imageData(rows * cols);

	// trova la massima magnitudine per normalizzare i valori
    float maxMagnitude = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (data[i][j] > maxMagnitude) {
                maxMagnitude = data[i][j];
            }
        }
    }

    // Normalizza le magnitudini e converte in byte
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            unsigned char pixelValue = static_cast<unsigned char>((data[i][j] / maxMagnitude) * 255);
            imageData[i * cols + j] = pixelValue;
        }
    }


    // Scrive il risultato come PNG
    if (stbi_write_png(filename, cols, rows, 1, imageData.data(), cols) == 0) {
        cerr << "Errore nella scrittura dell'immagine in " << filename << endl;
        throw runtime_error("Errore nella scrittura dell'immagine");
    }

    cout << "Immagine salvata come " << filename << endl;
}
int next_power_of_two(int N) {
    if (N <= 1) return 1; // Edge case for numbers <= 1
    int power = 1;
    while (power < N) {
        power <<= 1; // Equivalent to power = power * 2;
    }
    return power;
}