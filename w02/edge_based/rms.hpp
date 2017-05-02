#ifndef RMS_HPP
#define RMS_HPP
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <omp.h>

void rms_calc(const float* node_val, const float* node_old,
    const int nnode, const int i, const int node_dim=1){

    double rms=0;
    #pragma omp parallel for reduction(+:rms)
    for(int nodeIdx=0;nodeIdx<nnode;nodeIdx++){
      for(int dim = 0; dim<node_dim; dim++){
#ifdef USE_SOA
        int elementIdx = dim*nnode+nodeIdx;
#else
        int elementIdx = nodeIdx*node_dim+dim;
#endif
        rms+= (node_old[elementIdx]-node_val[elementIdx])*
          (node_old[elementIdx]-node_val[elementIdx]);
      }
    }
    rms = sqrt(rms/(nnode*node_dim));
    double max = *std::max_element(node_val,node_val+nnode);
    printf("%d\t%10.5e\tmax:%10.5e\n",i, rms, max);

}

#endif
