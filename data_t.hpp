#ifndef DATA_T_HPP
#define DATA_T_HPP
#include "helper_cuda.h"
#include <cassert>
#include <cuda.h>
#include <vector>

template <typename T> struct data_t {
private:
  // set_size: #elements, dim: #value per element,
  MY_SIZE size, dim;

  // data pointers
  T *data, *data_d;

public:
  // constructors
  data_t() : size(0), dim(0), data(nullptr), data_d(nullptr) {}
  data_t(MY_SIZE, MY_SIZE);

  data_t(const data_t &) = delete;
  data_t operator=(const data_t &) = delete;

  T &operator[](MY_SIZE ind);
  const T &operator[](MY_SIZE ind) const;
  T *begin();
  T *end();

  // functions to manage state between host and device memory
  void flushToHost();
  void flushToDevice();
  void initDeviceMemory();

  MY_SIZE getSize() const { return size; }
  MY_SIZE getDim() const { return dim; }
  T *getDeviceData() { return data_d; }

  // dtor
  ~data_t();
};

template <typename T> class device_data_t {
private:
  T *data_d;

public:
  explicit device_data_t(const std::vector<T> &host_buffer);
  device_data_t(const T *host_buffer, MY_SIZE length);
  ~device_data_t();

  device_data_t(const device_data_t &) = delete;
  device_data_t &operator=(const device_data_t &) = delete;
  device_data_t(device_data_t<T> &&);
  device_data_t &operator=(device_data_t<T> &&);

  operator T *() const { return data_d; }
};

template <bool Cond, typename T1, typename T2> struct choose_t {};

template <typename T1, typename T2> struct choose_t<true, T1, T2> {
  typedef T1 type;
  static type &&ret_value(T1 &&v, T2 &&) { return std::move(v); }
};

template <typename T1, typename T2> struct choose_t<false, T1, T2> {
  typedef T2 type;
  static type &&ret_value(T1 &&, T2 &&v) { return std::move(v); }
};

template <typename T>
data_t<T>::data_t(MY_SIZE set_s, MY_SIZE set_d)
    : size(set_s), dim(set_d), data(nullptr), data_d(nullptr) {
  data = new T[size * dim];
}

template <typename T> data_t<T>::~data_t() {
  delete[] data;
  if (data_d) {
    checkCudaErrors(cudaFree(data_d));
  }
}

template <typename T> T &data_t<T>::operator[](MY_SIZE ind) {
  return data[ind];
}

template <typename T> const T &data_t<T>::operator[](MY_SIZE ind) const {
  return data[ind];
}

template <typename T> void data_t<T>::flushToHost() {
  assert(data_d != nullptr);
  MY_SIZE bytes = size * dim * sizeof(T);
  checkCudaErrors(cudaMemcpy(data, data_d, bytes, cudaMemcpyDeviceToHost));
}

template <typename T> void data_t<T>::flushToDevice() {
  assert(data_d != nullptr);
  MY_SIZE bytes = size * dim * sizeof(T);
  checkCudaErrors(cudaMemcpy(data_d, data, bytes, cudaMemcpyHostToDevice));
}

template <typename T> void data_t<T>::initDeviceMemory() {
  assert(data_d == nullptr);
  int bytes = size * dim * sizeof(T);
  checkCudaErrors(cudaMalloc((void **)&data_d, bytes));
  checkCudaErrors(cudaMemcpy(data_d, data, bytes, cudaMemcpyHostToDevice));
}

template <typename T> T *data_t<T>::begin() { return data; }

template <typename T> T *data_t<T>::end() { return data + (size * dim); }

template <typename T>
device_data_t<T>::device_data_t(const std::vector<T> &host_buffer)
    : device_data_t(host_buffer.data(), host_buffer.size()) {}

template <typename T>
device_data_t<T>::device_data_t(const T *host_buffer, MY_SIZE length) {
  checkCudaErrors(cudaMalloc((void **)&data_d, sizeof(T) * length));
  checkCudaErrors(cudaMemcpy(data_d, host_buffer, sizeof(T) * length,
                             cudaMemcpyHostToDevice));
}

template <typename T> device_data_t<T>::~device_data_t() {
  checkCudaErrors(cudaFree(data_d));
}

template <typename T>
device_data_t<T>::device_data_t(device_data_t<T> &&other) {
  data_d = other.data_d;
  other.data_d = nullptr;
}

template <typename T>
device_data_t<T> &device_data_t<T>::operator=(device_data_t<T> &&rhs) {
  std::swap(data_d, rhs.data_d);
  return *this;
}

#endif /* end of guard DATA_T_HPP */
