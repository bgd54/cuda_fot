#!/bin/bash

cd ../w02/edge_based

mkdir output

make

for ver in edge_cpu edge_coloring edge_omp edge_gpu edge_gpu_2l edge_gpu_cw edge_gpu_ca
do
  params=("-dx 256 -dy 257" "-dx 1000 -dy 2000")
  names=(256 1000)
  for i in 0 1
  do 
    for bidir in "" "-bidir"
    do
      echo "./$ver ${params[$i]} $bidir"
      ./$ver ${params[$i]} $bidir > output/$ver${names[$i]}$bidir
    done
  done
done
