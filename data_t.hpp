#ifndef DATA_T_HPP
#define DATA_T_HPP
#include "helper_cuda.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cuda.h>
#include <vector>

struct data_t {
private:
  // set_size: #elements, dim: #value per element,
  MY_SIZE size, dim;
  unsigned type_size;

  // data pointers
  char *data, *data_d;

public:
  // constructors
  data_t(MY_SIZE set_s, MY_SIZE set_dim, unsigned type_size);
  template <typename T> static data_t create(MY_SIZE size, MY_SIZE dim) {
    return data_t(size, dim, sizeof(T));
  }

  data_t(const data_t &) = delete;
  data_t operator=(const data_t &) = delete;
  data_t(data_t &&);
  data_t &operator=(data_t &&);

  template <typename T = char> T &operator[](MY_SIZE ind);
  template <typename T = char> const T &operator[](MY_SIZE ind) const;
  template <typename T = char> T *begin();
  template <typename T = char> T *end();
  template <typename T = char> const T *cbegin() const;
  template <typename T = char> const T *cend() const;

  // functions to manage state between host and device memory
  void flushToHost();
  void flushToDevice();
  void initDeviceMemory();

  MY_SIZE getSize() const { return size; }
  unsigned getTypeSize() const { return type_size; }
  MY_SIZE getDim() const { return dim; }
  template <typename T = char> T *getDeviceData() {
    assert((std::is_same<char, T>::value || sizeof(T) == type_size));
    return reinterpret_cast<T *>(data_d);
  }

  // dtor
  ~data_t();
};

class device_data_t {
private:
  char *data_d;

public:
  // explicit device_data_t(const std::vector<T> &host_buffer);
  template <typename T>
  static device_data_t create(const std::vector<T> &host_buffer);
  device_data_t(const char *host_buffer, MY_SIZE length);
  ~device_data_t();

  device_data_t(const device_data_t &) = delete;
  device_data_t &operator=(const device_data_t &) = delete;
  device_data_t(device_data_t &&);
  device_data_t &operator=(device_data_t &&);

  template <typename T> operator T *() const {
    return reinterpret_cast<T *>(data_d);
  }
};

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

template <typename T> T &data_t::operator[](MY_SIZE ind) {
  return *reinterpret_cast<T *>(data + ind * type_size);
}

template <typename T> const T &data_t::operator[](MY_SIZE ind) const {
  return *reinterpret_cast<T *>(data + ind * type_size);
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
  assert(data_d == nullptr);
  MY_SIZE bytes = size * dim * type_size;
  checkCudaErrors(cudaMalloc((void **)&data_d, bytes));
  checkCudaErrors(cudaMemcpy(data_d, data, bytes, cudaMemcpyHostToDevice));
}

template <typename T> T *data_t::begin() { return reinterpret_cast<T *>(data); }

template <typename T> T *data_t::end() {
  return reinterpret_cast<T *>(data + (size * dim * type_size));
}

template <typename T> const T *data_t::cbegin() const {
  return reinterpret_cast<T *>(data);
}

template <typename T> const T *data_t::cend() const {
  return reinterpret_cast<T *>(data + (size * dim * type_size));
}

template <typename T>
device_data_t device_data_t::create(const std::vector<T> &host_buffer) {
  return device_data_t(reinterpret_cast<const char *>(host_buffer.data()),
                       host_buffer.size() * sizeof(T));
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

/* index {{{1 */
template <bool SOA = false>
__host__ __device__ __forceinline__ MY_SIZE index(MY_SIZE num_points,
                                                  MY_SIZE node_ind, MY_SIZE Dim,
                                                  MY_SIZE dim) {
  if (SOA) {
    return dim * num_points + node_ind;
  } else {
    return node_ind * Dim + dim;
  }
}
/* 1}}} */

template <bool SOA, typename UnsignedType>
void reorderData(data_t &point_data,
                 const std::vector<UnsignedType> &permutation) {
  std::vector<char> old_data(point_data.begin<char>(), point_data.end<char>());
  for (MY_SIZE i = 0; i < point_data.getSize(); ++i) {
    for (MY_SIZE d = 0; d < point_data.getDim(); ++d) {
      MY_SIZE old_ind =
          index<SOA>(point_data.getSize(), i, point_data.getDim(), d);
      MY_SIZE new_ind = index<SOA>(point_data.getSize(), permutation[i],
                                   point_data.getDim(), d);
      std::copy_n(old_data.begin() + old_ind * point_data.getTypeSize(),
                  point_data.getTypeSize(),
                  point_data.begin() + new_ind * point_data.getTypeSize());
    }
  }
}

template <bool SOA, typename T, typename UnsignedType>
void reorderData(std::vector<T> &point_data, MY_SIZE Dim,
                 const std::vector<UnsignedType> &permutation) {
  std::vector<T> old_data(point_data.begin(), point_data.end());
  for (MY_SIZE i = 0; i < point_data.size() / Dim; ++i) {
    for (MY_SIZE d = 0; d < Dim; ++d) {
      MY_SIZE old_ind = index<SOA>(point_data.size() / Dim, i, Dim, d);
      MY_SIZE new_ind =
          index<SOA>(point_data.size() / Dim, permutation[i], Dim, d);
      point_data[new_ind] = old_data[old_ind];
    }
  }
}

/**
 * Reorders data using the inverse of the permutation given
 */
template <bool SOA, typename UnsignedType>
void reorderDataInverse(data_t &point_data,
                        const std::vector<UnsignedType> &permutation) {
  std::vector<char> old_data(point_data.begin<char>(), point_data.end<char>());
  for (MY_SIZE i = 0; i < point_data.getSize(); ++i) {
    for (MY_SIZE d = 0; d < point_data.getDim(); ++d) {
      MY_SIZE old_ind = index<SOA>(point_data.getSize(), permutation[i],
                                   point_data.getDim(), d);
      MY_SIZE new_ind =
          index<SOA>(point_data.getSize(), i, point_data.getDim(), d);
      std::copy_n(old_data.begin() + old_ind * point_data.getTypeSize(),
                  point_data.getTypeSize(),
                  point_data.begin() + new_ind * point_data.getTypeSize());
    }
  }
}

/**
 * Reorders data using the inverse of the permutation given
 * Specialised for SOA structures, where the reordering is only on part of the
 * container (so the data corresponding to the different dimensions are not next
 * to each other)
 */
template <typename T, typename UnsignedType>
void reorderDataInverseVectorSOA(
    std::vector<typename std::vector<T>::iterator> point_data_begin,
    typename std::vector<T>::iterator point_data_end_first,
    const std::vector<UnsignedType> &permutation) {
  unsigned Dim = point_data_begin.size();
  assert(Dim >= 1);
  MY_SIZE size = std::distance(point_data_begin[0], point_data_end_first);
  assert(permutation.size() == size);
  std::vector<T> old_data(Dim * size);
  typename std::vector<T>::iterator it = old_data.begin();
  for (typename std::vector<T>::iterator ii : point_data_begin) {
    typename std::vector<T>::iterator end = std::next(ii, size);
    std::copy(ii, end, it);
    std::advance(it, size);
  }
  for (MY_SIZE i = 0; i < size; ++i) {
    for (MY_SIZE d = 0; d < Dim; ++d) {
      MY_SIZE old_ind = index<true>(size, permutation[i], Dim, d);
      *(point_data_begin[d]++) = old_data[old_ind];
    }
  }
}

template <typename UnsignedType>
void reorderDataInverseSOA(data_t &point_data, MY_SIZE from, MY_SIZE to,
                           const std::vector<UnsignedType> &permutation) {
  const MY_SIZE data_dim = point_data.getDim();
  assert(data_dim >= 1);
  const MY_SIZE size = to - from;
  const unsigned type_size = point_data.getTypeSize();
  assert(permutation.size() == size);
  data_t old_data(size, data_dim, type_size);
  for (MY_SIZE d = 0; d < data_dim; ++d) {
    std::copy_n(point_data.begin() +
                    type_size * (d * point_data.getSize() + from),
                type_size * size, old_data.begin() + type_size * d * size);
  }
  for (MY_SIZE i = 0; i < size; ++i) {
    for (MY_SIZE d = 0; d < data_dim; ++d) {
      MY_SIZE old_ind = index<true>(size, permutation[i], data_dim, d);
      MY_SIZE new_ind =
          index<true>(point_data.getSize(), from + i, data_dim, d);
      point_data[new_ind] = old_data[old_ind];
      std::copy_n(old_data.begin() + old_ind * type_size, type_size,
                  point_data.begin() + new_ind * type_size);
    }
  }
}

template <class Iterator>
inline void AOStoSOA(Iterator begin, Iterator end, unsigned Dim) {
  MY_SIZE size = std::distance(begin, end);
  assert(size % Dim == 0);
  MY_SIZE num_data = size / Dim;
  using DataType = typename std::iterator_traits<Iterator>::value_type;
  data_t tmp = data_t::create<DataType>(num_data, Dim);
  Iterator cur = begin;
  for (MY_SIZE i = 0; i < num_data; ++i) {
    for (MY_SIZE d = 0; d < Dim; ++d) {
      bool b = index<false>(num_data, i, Dim, d) == std::distance(begin, cur);
      assert(b);
      tmp.operator[]<DataType>(index<true>(num_data, i, Dim, d)) = *cur++;
    }
  }
  std::copy(tmp.begin<DataType>(), tmp.end<DataType>(), begin);
}

template <class T>
inline void AOStoSOA(std::vector<T> &container, unsigned Dim) {
  AOStoSOA<typename std::vector<T>::iterator>(container.begin(),
                                              container.end(), Dim);
}

template <class T> inline void AOStoSOA(data_t &container) {
  AOStoSOA<T *>(container.begin<T>(), container.end<T>(), container.getDim());
}

#endif /* end of guard DATA_T_HPP */
// vim:set et sw=2 ts=2 fdm=marker:
