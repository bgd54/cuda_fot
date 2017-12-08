#include "data_t.hpp"

data_t::data_t(MY_SIZE set_s, MY_SIZE set_dim, MY_SIZE type_size_)
    : size(set_s), dim(set_dim), type_size(type_size_), data(nullptr),
      data_d(nullptr) {
  data = new char[size * dim * type_size];
}

data_t::~data_t() {
  delete[] data;
  if (data_d) {
    checkCudaErrors(cudaFree(data_d));
  }
}

data_t::data_t(data_t &&other) {
  size = other.size;
  other.size = 0;
  dim = other.dim;
  other.dim = 0;
  type_size = other.type_size;
  other.type_size = 0;
  data = other.data;
  other.data = nullptr;
  data_d = other.data_d;
  other.data_d = nullptr;
}

data_t &data_t::operator=(data_t &&rhs) {
  std::swap(size, rhs.size);
  std::swap(dim, rhs.dim);
  std::swap(type_size, rhs.type_size);
  std::swap(data, rhs.data);
  std::swap(data_d, rhs.data_d);
  return *this;
}

void data_t::flushToHost() {
  assert(data_d != nullptr);
  MY_SIZE bytes = size * dim * type_size;
  checkCudaErrors(cudaMemcpy(data, data_d, bytes, cudaMemcpyDeviceToHost));
}

void data_t::flushToDevice() {
  assert(data_d != nullptr);
  MY_SIZE bytes = size * dim * type_size;
  checkCudaErrors(cudaMemcpy(data_d, data, bytes, cudaMemcpyHostToDevice));
}

void data_t::initDeviceMemory() {
  MY_SIZE bytes = size * dim * type_size;
  if (data_d == nullptr) {
    checkCudaErrors(cudaMalloc((void **)&data_d, bytes));
  }
  checkCudaErrors(cudaMemcpy(data_d, data, bytes, cudaMemcpyHostToDevice));
}

device_data_t::device_data_t(const char *host_buffer, MY_SIZE length) {
  checkCudaErrors(cudaMalloc((void **)&data_d, length));
  checkCudaErrors(
      cudaMemcpy(data_d, host_buffer, length, cudaMemcpyHostToDevice));
}

device_data_t::~device_data_t() { checkCudaErrors(cudaFree(data_d)); }

device_data_t::device_data_t(device_data_t &&other) {
  data_d = other.data_d;
  other.data_d = nullptr;
}

device_data_t &device_data_t::operator=(device_data_t &&rhs) {
  std::swap(data_d, rhs.data_d);
  return *this;
}

__global__ void _copyKernel(const float *__restrict__ a, float *__restrict__ b,
                            MY_SIZE size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const float4 *__restrict__ a_ = reinterpret_cast<const float4 *>(a);
  float4 *__restrict__ b_ = reinterpret_cast<float4 *>(b);
  if ((tid + 1) * 4 <= size) {
    b_[tid] = a_[tid];
  } else {
    for (MY_SIZE i = 0; i + tid * 4 < size; ++i) {
      b[4 * tid + i] = a[4 * tid + i];
    }
  }
}

void copyKernel(const float *a, float *b, MY_SIZE size, MY_SIZE num_blocks,
                MY_SIZE block_size) {
  _copyKernel<<<num_blocks, block_size>>>(a, b, size);
}

/* vim:set et sts=2g sw=4 ts=4 fdm=marker: */
