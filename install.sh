mkdir ./build
mkdir ./output

git clone https://github.com/ismrmrd/ismrmrd

cd ismrmrd || return 1
mkdir build
cd build || return 2
cmake ../
make
sudo make install

cd ../../build || return 3
cmake ..
cmake --build .
