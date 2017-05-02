#!/bin/bash

cd ../w02/edge_based

mkdir output

touch output/versions

#make

params="-bidir -dx 1600 -dy 1600"

for ver in edge_omp edge_gpu edge_gpu_2l edge_gpu_cw edge_gpu_ca edge_gpu_2l_r edge_gpu_cw_r edge_gpu_ca_r edge_omp_soa edge_gpu_soa edge_gpu_2l_soa edge_gpu_cw_soa edge_gpu_ca_soa edge_gpu_2l_r_soa edge_gpu_cw_r_soa edge_gpu_ca_r_soa
do
  mkdir output/${ver}
  for dim in 1 2 4 8 16 24 26 28 30
  do
    echo "./$ver ${params} -ndim ${dim}"
    ./$ver ${params} -ndim ${dim} | tee output/${ver}/${ver}_${dim}
    echo "${ver}_${dim}" >> output/versions
  done
done
