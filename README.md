# VectorSetSearch

## Install

faiss:

```
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -DBUILD_TESTING=OFF -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_INSTALL_PREFIX=$HOME/local/faiss -B build .
make -C build -j faiss
make -C build install
```

eigen:

```
wget https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz
tar -xvzf eigen-5.0.0.tar.gz
cd eigen-5.0.0
cmake -DCMAKE_INSTALL_PREFIX=$HOME/local/Eigen3 -B build .
make -C build install
```

pybind11:

```
git clone https://github.com/pybind/pybind11
cd pybind11
cmake -DCMAKE_INSTALL_PREFIX=$HOME/local/pybind11 -B build .
make -C build install
```

## Build & Run

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./vss_test 128 ms-marco/vectors-colbert/k10_s1K_v63K hnsw
```
