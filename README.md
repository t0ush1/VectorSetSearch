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
cmake -DCMAKE_INSTALL_PREFIX=$HOME/local/eigen -B build .
make -C build install
```

## Build & Run

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./vss_test 128 maxsim ms-marco/vectors-colbert/k10_s1K_v137K single_hnsw
./vss_test 768 dtw droid/vectors-dinov2/64-32-Uni_8_16-10-1K multi_hnsw
```
