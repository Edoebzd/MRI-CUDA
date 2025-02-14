# MRI-CUDA

Cuda-accelerated processing of raw MRI data (coming from [mridata.org](https://mridata.org)) to generate output images in grayscale.

### Data requirements
- Slices must be square (width = height)
- Minimum slice width = 9px
- Maximum slice width = 1024px
- Recommended slice width = 257-1024px
- Output images width is equal to slice width rounded up to the next power of 2.
- Multiple slices per dataset supported
- Multiple channels per slice supported
- Maximum total amount of raw data depends on GPU and CPU memory size (minimum memory required (CPU and GPU) = (8\*size\*size\*num_channels\*num_slices) bytes)

## How to build (Linux)
### Dependencies required
- git
- CUDAToolkit
- CMake (latest version)
- libpugixml
- ISMRMRD lib (installed automatically by the install script)

#### How to install CMake latest
Install latest version using snap
```
sudo apt update
sudo apt install snapd
sudo snap install cmake --classic
```
Installed by default in `/snap/bin`, needs to manually be included in PATH env.

#### How to install libpugixml
```
sudo apt install libpugixml-dev
```

### How to compile the ISMRMRD library and the MRI-CUDA project
Run the script `install.sh`
> In case of errors about other missing libraries, install them manually using apt or other package managers.

## How to run
`./build/MRI-CUDA <inputFile> <outputDir>`

## Example run
An example is provided as "ready-to-use".
The required data can be downloaded by running the script `exampleData.sh` (Around 1.6 Gb).
The project can then be executed by running the script `exampleRun.sh` which is already preconfigured.

## Developed by
Edoardo Bosi
Alfredo Grande

#### Tested successfully with
```
CPU:
Intel(R) Xeon(R) CPU @ 2.20GHz (x4)
15 Gb RAM

GPU: 
Nvidia Tesla T4 (16 Gb)

```