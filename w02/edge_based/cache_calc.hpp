#ifndef CACHE_MAP_HPP
#define CACHE_MAP_HPP
#include "coloring.hpp"
#include <algorithm>
#include <set>
#include <vector>
#include "helper_cuda.h"
#include "helper_string.h"

struct cacheMap{
  #define MY_IDX_TYPE unsigned short
  int numblock, nedge;
  int* globalToCacheMap;
  int* blockOffsets;
  MY_IDX_TYPE *readC, *writeC;
  //____________Device pointers______________
  int *globalToCacheMap_d, *blockOffsets_d;
  MY_IDX_TYPE *readC_d, *writeC_d;

  cacheMap(int _numb, int* _globalToC, int* _blockOffs, 
      MY_IDX_TYPE* wc, MY_IDX_TYPE* rc, int _nedge):numblock(_numb), 
      nedge(_nedge), globalToCacheMap(_globalToC), blockOffsets(_blockOffs), 
      readC(rc), writeC(wc) {

    checkCudaErrors( cudaMalloc((void**)&blockOffsets_d,
          numblock*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**)&globalToCacheMap_d,
          blockOffsets[numblock-1]*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**)&writeC_d,
          nedge*sizeof(MY_IDX_TYPE)) );
    checkCudaErrors( cudaMalloc((void**)&readC_d,
          nedge*sizeof(MY_IDX_TYPE)) );

    checkCudaErrors( cudaMemcpy(blockOffsets_d, blockOffsets,
          numblock*sizeof(int),  cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(globalToCacheMap_d, globalToCacheMap,
          blockOffsets[numblock-1]*sizeof(int),  cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(writeC_d, writeC,
          nedge*sizeof(MY_IDX_TYPE),  cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(readC_d, readC,
          nedge*sizeof(MY_IDX_TYPE),  cudaMemcpyHostToDevice) );
        
  }

  ~cacheMap(){
    free(globalToCacheMap);
    free(blockOffsets);
    free(readC);
    free(writeC);

    //cuda freee
    checkCudaErrors( cudaFree(globalToCacheMap_d) );
    checkCudaErrors( cudaFree(blockOffsets_d) );
    checkCudaErrors( cudaFree(writeC_d) );
    checkCudaErrors( cudaFree(readC_d) );
  }

};

cacheMap genCacheMap(const int* enode, const int &nedge, 
    const Block_coloring & bc){

  int maxc=0, minc=bc.bs, sumc = 0; // helper variables for recycling faktor
  
  MY_IDX_TYPE * readC = (MY_IDX_TYPE*) malloc(nedge*sizeof(MY_IDX_TYPE));
  MY_IDX_TYPE * writeC = (MY_IDX_TYPE*) malloc(nedge*sizeof(MY_IDX_TYPE));

  vector<vector<int>> pointsToCachePerBlocks(bc.numblock);

#pragma omp parallel for reduction(+:sumc) reduction(max:maxc) \
  reduction(min:minc)
  for(int bIdx=0; bIdx < bc.numblock; ++bIdx){
    int start= bIdx*bc.bs, end= std::min((bIdx+1)*bc.bs,nedge);
    std::set<int> pointsToCache;
    for(int tid=0; tid + start < end; ++tid){
      int Idx = tid+start;
      writeC[Idx] = enode[2*bc.color_reord[Idx] + 1]; 
      readC[Idx]  = enode[2*bc.color_reord[Idx] + 0];
      pointsToCache.insert(writeC[Idx]);
      pointsToCache.insert(readC[Idx]);
    }
    int cache_size = pointsToCache.size();
    if(minc>cache_size) minc = cache_size;
    else if(cache_size>maxc) maxc = cache_size;

    sumc += cache_size;

    std::copy( pointsToCache.begin(), pointsToCache.end(),
        std::back_insert_iterator<vector<int>>(pointsToCachePerBlocks[bIdx]) );
    #pragma omp parallel for
    for(int Idx = start; Idx<end; ++Idx){
      writeC[Idx] = *std::find(pointsToCachePerBlocks[bIdx].begin(),
          pointsToCachePerBlocks[bIdx].end(), writeC[Idx]);
      readC[Idx] = *std::find(pointsToCachePerBlocks[bIdx].begin(),
          pointsToCachePerBlocks[bIdx].end(), readC[Idx]);
    }
  }
  printf(
      "cache recycling factor:\nworst block: %lf\tbest block: %lf\t avg:%lf\n\n",
        maxc/(double)bc.bs,((double)minc)/bc.bs, sumc/(double)nedge );

  int* blockOffsets = (int*) malloc(bc.numblock*sizeof(int));
  blockOffsets[0] = pointsToCachePerBlocks[0].size();
  for(int bIdx=1; bIdx < bc.numblock; ++bIdx) 
    blockOffsets[bIdx] = 
      blockOffsets[bIdx-1] + pointsToCachePerBlocks[bIdx].size();
  int* globalToCacheMap = 
    (int*) malloc(blockOffsets[bc.numblock-1]*sizeof(int));
  
#pragma omp parallel for
  for(int bIdx=0; bIdx < bc.numblock; ++bIdx){
    for(size_t i=0; i<pointsToCachePerBlocks[bIdx].size();++i){
      globalToCacheMap[(bIdx>0?blockOffsets[bIdx-1]:0)+i] = 
        pointsToCachePerBlocks[bIdx][i]; 
    }
  }

  return cacheMap(bc.numblock, globalToCacheMap, blockOffsets, readC,
                  writeC, nedge);
}


#endif /* end of guard CACHE_MAP_HPP */
