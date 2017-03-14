#ifndef RMS_HPP
#define RMS_HPP
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <omp.h>

void rms_calc(const float* node_val, const float* node_old,
    const int nnode, const int i){

    double rms=0;
    #pragma omp parallel for reduction(+:rms)
    for(int nodeIdx=0;nodeIdx<nnode;nodeIdx++){
      rms+= (node_old[nodeIdx]-node_val[nodeIdx])*
        (node_old[nodeIdx]-node_val[nodeIdx]);
    }
    rms = sqrt(rms/nnode);
    double max = *std::max_element(node_val,node_val+nnode);
    printf("%d\t%10.5e\tmax:%10.5e\n",i, rms, max);

}

#endif
