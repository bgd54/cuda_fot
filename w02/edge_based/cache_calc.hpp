#ifndef CACHE_MAP_HPP
#define CACHE_MAP_HPP
#include "coloring.hpp"
#include <algorithm>
#include <set>
#include <vector>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "helper_cuda.h"
#include "helper_string.h"
#endif

struct cacheMap{
  #define MY_IDX_TYPE unsigned short
  int numblock, nedge;
  int* globalToCacheMap;
  int* blockOffsets;
  int* globalReadToCacheMap;
  int* blockReadOffsets;
  MY_IDX_TYPE *readC, *writeC;
  #ifdef USE_CUDA
  //____________Device pointers______________
  int *globalToCacheMap_d, *blockOffsets_d;
  int *globalReadToCacheMap_d, *blockReadOffsets_d;
  MY_IDX_TYPE *readC_d, *writeC_d;
  #endif
  //Constructors
  cacheMap(int _numb, int* _globalToC, int* _blockOffs,  int* _globalRToC,
      int* _blockROffs,  MY_IDX_TYPE* rc, MY_IDX_TYPE* wc, int _nedge);

  ~cacheMap();

};

cacheMap genCacheMap(const int* enode, const int &nedge, 
    const Block_coloring & bc);

#endif /* end of guard CACHE_MAP_HPP */
