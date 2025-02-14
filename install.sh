mkdir ./build
mkdir ./output
mkdir ./ismrmrd-install

git clone https://github.com/ismrmrd/ismrmrd

cd ismrmrd || return 1
mkdir build
cd build || return 2
cmake -DCMAKE_INSTALL_PREFIX=./ismrmrd-install/ ../
make
make install

dir=$(pwd)
cd ../../build || return 3
cmake -DCMAKE_BUILD_TYPE=Release "-DISMRMRD_DIR=${dir}/ismrmrd-install/lib/cmake/ISMRMRD" ..
cmake --build .
