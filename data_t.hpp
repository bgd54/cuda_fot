#ifndef DATA_T_HPP
#define DATA_T_HPP
#include "helper_cuda.h"
#include <array>
#include <cassert>
#include <cuda.h>
#include <vector>

template <typename T, unsigned Dim> struct data_t {
private:
  // set_size: #elements, dim: #value per element,
  MY_SIZE size;

  // data pointers
  T *data, *data_d;

public:
  static constexpr unsigned dim = Dim;

public:
  // constructors
  data_t() : size(0), data(nullptr), data_d(nullptr) {}
  data_t(MY_SIZE);

  data_t(const data_t &) = delete;
  data_t operator=(const data_t &) = delete;
  data_t(data_t &&);
  data_t &operator=(data_t &&);

  T &operator[](MY_SIZE ind);
  const T &operator[](MY_SIZE ind) const;
  T *begin();
  T *end();
  const T *cbegin() const;
  const T *cend() const;

  // functions to manage state between host and device memory
  void flushToHost();
  void flushToDevice();
  void initDeviceMemory();

  MY_SIZE getSize() const { return size; }
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

template <typename T, unsigned Dim>
data_t<T, Dim>::data_t(MY_SIZE set_s)
    : size(set_s), data(nullptr), data_d(nullptr) {
  data = new T[size * dim];
}

template <typename T, unsigned Dim> data_t<T, Dim>::~data_t() {
  delete[] data;
  if (data_d) {
    checkCudaErrors(cudaFree(data_d));
  }
}

template <typename T, unsigned Dim>
data_t<T, Dim>::data_t(data_t<T, Dim> &&other) {
  size = other.size;
  other.size = 0;
  data = other.data;
  other.data = nullptr;
  data_d = other.data_d;
  other.data_d = nullptr;
}

template <typename T, unsigned Dim>
data_t<T, Dim> &data_t<T, Dim>::operator=(data_t<T, Dim> &&rhs) {
  std::swap(size, rhs.size);
  std::swap(data, rhs.data);
  std::swap(data_d, rhs.data_d);
  return *this;
}

template <typename T, unsigned Dim> T &data_t<T, Dim>::operator[](MY_SIZE ind) {
  return data[ind];
}

template <typename T, unsigned Dim>
const T &data_t<T, Dim>::operator[](MY_SIZE ind) const {
  return data[ind];
}

template <typename T, unsigned Dim> void data_t<T, Dim>::flushToHost() {
  assert(data_d != nullptr);
  MY_SIZE bytes = size * dim * sizeof(T);
  checkCudaErrors(cudaMemcpy(data, data_d, bytes, cudaMemcpyDeviceToHost));
}

template <typename T, unsigned Dim> void data_t<T, Dim>::flushToDevice() {
  assert(data_d != nullptr);
  MY_SIZE bytes = size * dim * sizeof(T);
  checkCudaErrors(cudaMemcpy(data_d, data, bytes, cudaMemcpyHostToDevice));
}

template <typename T, unsigned Dim> void data_t<T, Dim>::initDeviceMemory() {
  assert(data_d == nullptr);
  MY_SIZE bytes = size * dim * sizeof(T);
  checkCudaErrors(cudaMalloc((void **)&data_d, bytes));
  checkCudaErrors(cudaMemcpy(data_d, data, bytes, cudaMemcpyHostToDevice));
}

template <typename T, unsigned Dim> T *data_t<T, Dim>::begin() { return data; }

template <typename T, unsigned Dim> T *data_t<T, Dim>::end() {
  return data + (size * dim);
}

template <typename T, unsigned Dim> const T *data_t<T, Dim>::cbegin() const {
  return data;
}

template <typename T, unsigned Dim> const T *data_t<T, Dim>::cend() const {
  return data + (size * dim);
}

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

/* index {{{1 */
template <unsigned Dim = 1, bool SOA = false>
__host__ __device__ __forceinline__ MY_SIZE index(MY_SIZE num_points,
                                                  MY_SIZE node_ind,
                                                  MY_SIZE dim) {
  if (SOA) {
    return dim * num_points + node_ind;
  } else {
    return node_ind * Dim + dim;
  }
}
/* 1}}} */

template <unsigned Dim, bool SOA, typename T, typename UnsignedType>
void reorderData(data_t<T, Dim> &point_data,
                 const std::vector<UnsignedType> &permutation) {
  std::vector<T> old_data(point_data.begin(), point_data.end());
  for (MY_SIZE i = 0; i < point_data.getSize(); ++i) {
    for (MY_SIZE d = 0; d < Dim; ++d) {
      MY_SIZE old_ind = index<Dim, SOA>(point_data.getSize(), i, d);
      MY_SIZE new_ind =
          index<Dim, SOA>(point_data.getSize(), permutation[i], d);
      point_data[new_ind] = old_data[old_ind];
    }
  }
}

template <unsigned Dim, bool SOA, typename T, typename UnsignedType>
void reorderData(std::vector<T> &point_data,
                 const std::vector<UnsignedType> &permutation) {
  std::vector<T> old_data(point_data.begin(), point_data.end());
  for (MY_SIZE i = 0; i < point_data.size() / Dim; ++i) {
    for (MY_SIZE d = 0; d < Dim; ++d) {
      MY_SIZE old_ind = index<Dim, SOA>(point_data.size() / Dim, i, d);
      MY_SIZE new_ind =
          index<Dim, SOA>(point_data.getSize() / Dim, permutation[i], d);
      point_data[new_ind] = old_data[old_ind];
    }
  }
}



/**
 * Reorders data using the inverse of the permutation given
 */
template <unsigned Dim, bool SOA, typename T, typename UnsignedType>
void reorderDataInverse(data_t<T, Dim> &point_data,
                        const std::vector<UnsignedType> &permutation) {
  std::vector<T> old_data(point_data.begin(), point_data.end());
  for (MY_SIZE i = 0; i < point_data.getSize(); ++i) {
    for (MY_SIZE d = 0; d < Dim; ++d) {
      MY_SIZE old_ind =
          index<Dim, SOA>(point_data.getSize(), permutation[i], d);
      MY_SIZE new_ind = index<Dim, SOA>(point_data.getSize(), i, d);
      point_data[new_ind] = old_data[old_ind];
    }
  }
}

/**
 * Reorders data using the inverse of the permutation given
 * Specialised for SOA structures, where the reordering is only on part of the
 * container (so the data corresponding to the different dimensions are not next
 * to each other)
 */
template <unsigned Dim, typename T, typename UnsignedType>
void reorderDataInverseVectorSOA(
    std::array<typename std::vector<T>::iterator, Dim> point_data_begin,
    typename std::vector<T>::iterator point_data_end_first,
    const std::vector<UnsignedType> &permutation) {
  static_assert(Dim >= 1, "reorderDataInverseVectorSOA called with Dim < 1");
  MY_SIZE size = std::distance(point_data_begin[0], point_data_end_first);
  std::vector<T> old_data(Dim * size);
  typename std::vector<T>::iterator it = old_data.begin();
  for (typename std::vector<T>::iterator ii : point_data_begin) {
    typename std::vector<T>::iterator end = std::next(ii, size);
    std::copy(ii, end, it);
    std::advance(it, size);
  }
  for (MY_SIZE i = 0; i < size; ++i) {
    for (MY_SIZE d = 0; d < Dim; ++d) {
      MY_SIZE old_ind = index<Dim, true>(size, permutation[i], d);
      *point_data_begin[d]++ = old_data[old_ind];
    }
  }
}

template <unsigned Dim, class Iterator>
inline void AOStoSOA(Iterator begin, Iterator end) {
  MY_SIZE size = std::distance(begin, end);
  assert(size % Dim == 0);
  MY_SIZE num_data = size / Dim;
  using DataType = typename std::iterator_traits<Iterator>::value_type;
  data_t<DataType, Dim> tmp(num_data);
  Iterator cur = begin;
  for (MY_SIZE i = 0; i < num_data; ++i) {
    for (MY_SIZE d = 0; d < Dim; ++d) {
      bool b = index<Dim, false>(num_data, i, d) == std::distance(begin, cur);
      assert(b);
      tmp[index<Dim, true>(num_data, i, d)] = *cur++;
    }
  }
  std::copy(tmp.begin(), tmp.end(), begin);
}

template <unsigned Dim, class T>
inline void AOStoSOA(std::vector<T> &container) {
  AOStoSOA<Dim>(container.begin(), container.end());
}

template <unsigned Dim, class T>
inline void AOStoSOA(data_t<T, Dim> &container) {
  AOStoSOA<Dim>(container.begin(), container.end());
}

#endif /* end of guard DATA_T_HPP */
// vim:set et sw=2 ts=2 fdm=marker:
