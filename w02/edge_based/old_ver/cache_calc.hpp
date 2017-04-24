#ifndef CACHE_MAP_HPP
#define CACHE_MAP_HPP
#include "coloring.hpp"
#include <algorithm>
#include <set>
#include <vector>
#ifdef USE_CUDA
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
      int* _blockROffs,  MY_IDX_TYPE* rc, MY_IDX_TYPE* wc, int _nedge):
    numblock(_numb), nedge(_nedge), globalToCacheMap(_globalToC),
    blockOffsets(_blockOffs), globalReadToCacheMap(_globalRToC),
    blockReadOffsets(_blockROffs), readC(rc), writeC(wc) {

  #ifdef USE_CUDA
    checkCudaErrors( cudaMalloc((void**)&blockOffsets_d,
          numblock*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**)&globalToCacheMap_d,
          blockOffsets[numblock-1]*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**)&blockReadOffsets_d,
          numblock*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**)&globalReadToCacheMap_d,
          blockReadOffsets[numblock-1]*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**)&writeC_d,
          nedge*sizeof(MY_IDX_TYPE)) );
    checkCudaErrors( cudaMalloc((void**)&readC_d,
          nedge*sizeof(MY_IDX_TYPE)) );

    checkCudaErrors( cudaMemcpy(blockOffsets_d, blockOffsets,
          numblock*sizeof(int),  cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(globalToCacheMap_d, globalToCacheMap,
          blockOffsets[numblock-1]*sizeof(int),  cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(blockReadOffsets_d, blockReadOffsets,
          numblock*sizeof(int),  cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(globalReadToCacheMap_d, globalReadToCacheMap,
          blockReadOffsets[numblock-1]*sizeof(int),  cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(writeC_d, writeC,
          nedge*sizeof(MY_IDX_TYPE),  cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(readC_d, readC,
          nedge*sizeof(MY_IDX_TYPE),  cudaMemcpyHostToDevice) );
  #endif      
  }

  ~cacheMap(){
    free(globalToCacheMap);
    free(blockOffsets);
    free(globalReadToCacheMap);
    free(blockReadOffsets);
    free(readC);
    free(writeC);

  #ifdef USE_CUDA
    //cuda freee
    checkCudaErrors( cudaFree(globalToCacheMap_d) );
    checkCudaErrors( cudaFree(blockOffsets_d) );
    checkCudaErrors( cudaFree(globalReadToCacheMap_d) );
    checkCudaErrors( cudaFree(blockReadOffsets_d) );
    checkCudaErrors( cudaFree(writeC_d) );
    checkCudaErrors( cudaFree(readC_d) );
  #endif
  }

};

cacheMap genCacheMap(const int* enode, const int &nedge, 
    const Block_coloring & bc){
  
  if(bc.reordcolor == nullptr){
    return cacheMap(0, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nedge);
  }

  int maxc=0, minc=bc.bs, sumc = 0; // helper variables for recycling faktor
  
  MY_IDX_TYPE * readC = (MY_IDX_TYPE*) malloc(nedge*sizeof(MY_IDX_TYPE));
  MY_IDX_TYPE * writeC = (MY_IDX_TYPE*) malloc(nedge*sizeof(MY_IDX_TYPE));
  //tmp vars for collecting written node idx per edge
  std::vector<int> _readC(nedge,-1);
  std::vector<int> _writeC(nedge,-1);

  vector<vector<int>> pointsToCachePerBlocks(bc.numblock);
  vector<vector<int>> readPointsToCachePerBlocks(bc.numblock);

#pragma omp parallel for reduction(+:sumc) reduction(max:maxc) \
  reduction(min:minc)
  for(int bIdx=0; bIdx < bc.numblock; ++bIdx){
    int start= bIdx*bc.bs, end= std::min((bIdx+1)*bc.bs,nedge);
    std::set<int> pointsToCache;
    std::set<int> readPointsToCache;
    for(int tid=0; tid + start < end; ++tid){
      int Idx = tid+start;
      _writeC[Idx] = enode[2*bc.color_reord[Idx] + 1]; 
      _readC[Idx]  = enode[2*bc.color_reord[Idx] + 0];
      pointsToCache.insert(_writeC[Idx]);
      readPointsToCache.insert(_readC[Idx]);
    }
    int cache_size = pointsToCache.size()+readPointsToCache.size();
    if(minc>cache_size) minc = cache_size;
    else if(cache_size>maxc) maxc = cache_size;

    sumc += cache_size;

    std::copy( pointsToCache.begin(), pointsToCache.end(),
        std::back_insert_iterator<vector<int>>(pointsToCachePerBlocks[bIdx]) );
    std::copy( readPointsToCache.begin(), readPointsToCache.end(),
        std::back_insert_iterator<vector<int>>(
          readPointsToCachePerBlocks[bIdx]) );

    #pragma omp parallel for
    for(int Idx = start; Idx<end; ++Idx){
      writeC[Idx] = (std::find(pointsToCachePerBlocks[bIdx].begin(),
          pointsToCachePerBlocks[bIdx].end(), _writeC[Idx]) - 
          pointsToCachePerBlocks[bIdx].begin());
      
      readC[Idx] = (std::find(readPointsToCachePerBlocks[bIdx].begin(),
          readPointsToCachePerBlocks[bIdx].end(), _readC[Idx]) - 
            readPointsToCachePerBlocks[bIdx].begin() );
    }
  }
  printf(
      "cache recycling factor:\nworst block: %lf\tbest block: %lf\t avg:%lf\n\n",
        maxc/(double)bc.bs,((double)minc)/bc.bs, sumc/(double)nedge );

  int* blockOffsets = (int*) malloc(bc.numblock*sizeof(int));
  int* blockReadOffsets = (int*) malloc(bc.numblock*sizeof(int));
  blockOffsets[0] = pointsToCachePerBlocks[0].size();
  blockReadOffsets[0] = readPointsToCachePerBlocks[0].size();
  
  for(int bIdx=1; bIdx < bc.numblock; ++bIdx){ 
    blockOffsets[bIdx] = 
      blockOffsets[bIdx-1] + pointsToCachePerBlocks[bIdx].size();
    blockReadOffsets[bIdx] = 
      blockReadOffsets[bIdx-1] + readPointsToCachePerBlocks[bIdx].size(); 
  }

  int* globalToCacheMap = 
    (int*) malloc(blockOffsets[bc.numblock-1]*sizeof(int));
  
  #pragma omp parallel for
  for(int bIdx=0; bIdx < bc.numblock; ++bIdx){
    for(size_t i=0; i<pointsToCachePerBlocks[bIdx].size();++i){
      globalToCacheMap[(bIdx>0?blockOffsets[bIdx-1]:0)+i] = 
        pointsToCachePerBlocks[bIdx][i]; 
    }
  }

  int* globalReadToCacheMap = 
    (int*) malloc(blockReadOffsets[bc.numblock-1]*sizeof(int));

  #pragma omp parallel for
  for(int bIdx=0; bIdx < bc.numblock; ++bIdx){
    for(size_t i=0; i<readPointsToCachePerBlocks[bIdx].size();++i){
      globalReadToCacheMap[(bIdx>0?blockReadOffsets[bIdx-1]:0)+i] = 
        readPointsToCachePerBlocks[bIdx][i]; 
    }
  }

  return cacheMap(bc.numblock, globalToCacheMap, blockOffsets,
      globalReadToCacheMap, blockReadOffsets, readC, writeC, nedge);
}


#endif /* end of guard CACHE_MAP_HPP */
