#ifndef DATA_T_HPP
#define DATA_T_HPP
#include <cassert>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "helper_cuda.h"
#include "helper_string.h"

template <typename T>
struct data_t{
private:
  //set_size: #elements, dim: #value per element,
  MY_SIZE size, dim;

  //data pointers
  T *data, *data_d;
public:
  //constructors
  data_t(): size(0), dim(0), data(nullptr), data_d(nullptr){}
  data_t(MY_SIZE, MY_SIZE);

  data_t(const data_t&)=delete;
  data_t operator=(const data_t&)=delete;

  T& operator[](MY_SIZE ind);
  const T& operator[](MY_SIZE ind) const;
  T* begin();
  T* end();

  //functions to manage state between host and device memory
  void flush_to_host();
  void flush_to_device();
  void create_device_memory();

  MY_SIZE getSize();
  MY_SIZE getDim();
  T* getData_d();


  //dtor
  ~data_t();
};

template <typename T>
data_t<T>::data_t(MY_SIZE set_s, MY_SIZE set_d):
  size(set_s), dim(set_d), data(nullptr), data_d(nullptr){
  data = new T[size*dim];  
}

template <typename T>
data_t<T>::~data_t(){ 
  delete[] data;
  if(data_d)
      checkCudaErrors( cudaFree(data_d) );
}

template <typename T>
T& data_t<T>::operator[](MY_SIZE ind){
  return data[ind];
}

template <typename T>
const T& data_t<T>::operator[](MY_SIZE ind)const{
  return data[ind];
}


template <typename T>
void data_t<T>::flush_to_host(){
  assert(data_d != nullptr);
  MY_SIZE bytes = size * dim * sizeof(T);
  checkCudaErrors( 
        cudaMemcpy(data, data_d, bytes,  cudaMemcpyDeviceToHost) );

}

template <typename T>
void data_t<T>::flush_to_device(){
  assert(data_d != nullptr);
  MY_SIZE bytes = size * dim * sizeof(T);
  checkCudaErrors( 
        cudaMemcpy(data_d, data, bytes,  cudaMemcpyHostToDevice) );

}

template <typename T>
void data_t<T>::create_device_memory(){
  assert(data_d == nullptr);
  int bytes = size * dim * sizeof(T);
  checkCudaErrors( cudaMalloc((void**)&data_d, bytes) );
  checkCudaErrors(
      cudaMemcpy(data_d, data, bytes,  cudaMemcpyHostToDevice) );
}

template <typename T>
MY_SIZE data_t<T>::getSize() { return size; }

template <typename T>
MY_SIZE data_t<T>::getDim() { return dim; }

template <typename T>
T* data_t<T>::getData_d() { return data_d; }

template <typename T>
T* data_t<T>::begin() { return data; }

template <typename T>
T* data_t<T>::end() { return data + ( size * dim ); }

#endif /* end of guard DATA_T_HPP */
