#include "graph.hpp"
#include "problem.hpp"
#include "timer.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

std::vector<MY_SIZE> histogram(const std::vector<idx_t> &partition_vector) {
  assert(partition_vector.size());
  MY_SIZE mx =
      *std::max_element(partition_vector.begin(), partition_vector.end());
  assert(*std::min_element(partition_vector.begin(), partition_vector.end()) ==
         0);
  std::vector<MY_SIZE> result(mx + 1, 0);
  for (idx_t i : partition_vector) {
    ++result[i];
  }
  return result;
}

std::pair<MY_SIZE, MY_SIZE> getMinMax(const std::vector<MY_SIZE> &v) {
  return std::make_pair(*std::min_element(v.begin(), v.end()),
                        *std::max_element(v.begin(), v.end()));
}

int measurePartitioning() {
  constexpr MY_SIZE ITER_NUM = 500;
  constexpr bool SOA = false;
  constexpr unsigned Dim = 8;
  {
    std::cout << "No partitioning:" << std::endl;
    /*std::ifstream f ("/data/mgiles/asulyok/rotor37_nonrenum");*/
    /*std::ifstream f("/data/mgiles/asulyok/grid_2049x2049_row_major");*/
    std::ifstream f("/data/mgiles/asulyok/grid_1153x1153_row_major");
    /*std::ifstream f("/data/mgiles/asulyok/grid_1025x1025_row_major");*/
    /*std::ifstream f ("/data/mgiles/asulyok/wave");*/
    Problem<Dim, Dim, SOA, float> problem(f, 288);
    /*problem.loopGPUHierarchical(ITER_NUM);*/
  }
  std::cout << "=========================================" << std::endl;
  /*{*/
  /*  std::cout << "No partitioning, 9x8:" << std::endl;*/
  /*  Problem<Dim, Dim, SOA, float> problem({1153,1153},{9,8});*/
  /*  problem.loopGPUHierarchical(ITER_NUM);*/
  /*}*/
  std::cout << "=========================================" << std::endl;
  {
    std::cout << "No partitioning, GPS:" << std::endl;
    /*std::ifstream f ("/data/mgiles/asulyok/rotor37_nonrenum.gps2");*/
    /*std::ifstream f("/data/mgiles/asulyok/grid_2049x2049_row_major.gps2");*/
    /*std::ifstream f("/data/mgiles/asulyok/grid_1025x1025_--squared.gps2");*/
    std::ifstream f("/data/mgiles/asulyok/grid_1153x1153_row_major.gps2");
    /*std::ifstream f ("/data/mgiles/asulyok/wave.gps2");*/
    Problem<Dim, Dim, SOA, float> problem(f, 288);
    /*problem.loopGPUHierarchical(ITER_NUM);*/
  }
  std::vector<real_t> tolerances = {1.001 /*, 1.005, 1.01, 1.05, */
                                    /*1.08, 1.09, 1.1, 1.11, 1.12, 1.15, 1.2*/};
  std::vector<std::array<idx_t, METIS_NOPTIONS>> options(12);
  std::vector<std::string> option_names(options.size());
  for (auto &option : options) {
    METIS_SetDefaultOptions(option.data());
    option[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
  }
  option_names[0] = "default";
  options[1][METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
  option_names[1] = "objective:min_communication";
  options[2][METIS_OPTION_CTYPE] = METIS_CTYPE_RM;
  option_names[2] = "random_matching";
  options[3][METIS_OPTION_IPTYPE] = METIS_IPTYPE_RANDOM;
  option_names[3] = "iptype:random";
  options[4][METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
  option_names[4] = "iptype:grow";
  options[5][METIS_OPTION_IPTYPE] = METIS_IPTYPE_EDGE;
  option_names[5] = "iptype:edge";
  options[6][METIS_OPTION_IPTYPE] = METIS_IPTYPE_NODE;
  option_names[6] = "iptype:node";
  options[7][METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
  option_names[7] = "rtype:fm";
  options[8][METIS_OPTION_RTYPE] = METIS_RTYPE_SEP1SIDED;
  option_names[8] = "rtype:sep1sided";
  options[9][METIS_OPTION_RTYPE] = METIS_RTYPE_SEP2SIDED;
  option_names[9] = "rtype:sep2sided";
  options[10][METIS_OPTION_NO2HOP] = 1;
  option_names[10] = "no2hop";
  options[11][METIS_OPTION_CONTIG] = 1;
  option_names[11] = "contiguous";
  for (std::size_t i = 0; i < tolerances.size(); ++i) {
    for (std::size_t j = 0; j < 1 /*options.size()*/; ++j) {
      /*std::ifstream f ("/data/mgiles/asulyok/rotor37_nonrenum");*/
      /*std::ifstream f("/data/mgiles/asulyok/grid_2049x2049_row_major.gps2");*/
      /*std::ifstream f("/data/mgiles/asulyok/grid_129x129x129_row_major");*/
      std::ifstream f("/data/mgiles/asulyok/grid_1153x1153_row_major.gps2");
      /*std::ifstream f ("/data/mgiles/asulyok/wave.gps");*/
      Problem<Dim, Dim, SOA, float> problem(f, 288);
      MY_SIZE block_size =
          std::floor(static_cast<double>(problem.block_size) / tolerances[i]);
      real_t tolerance = static_cast<double>(problem.block_size) / block_size;
      std::cout << "=========================================" << std::endl
                << "Tolerance: " << tolerance << std::endl;
      std::cout << "Option: " << option_names[j] << std::endl;
      /*Timer t;*/
      /*std::vector<idx_t> p =*/
      /*    partitionMetis(problem.graph.getLineGraph(), block_size,
       * tolerance,*/
      /*                   options[j].data());*/
      /*auto tt = t.getTime();*/
      /*auto v = histogram(p);*/
      /*std::pair<MY_SIZE, MY_SIZE> minmax = getMinMax(v);*/
      /*auto mean =*/
      /*    static_cast<double>(std::accumulate(v.begin(), v.end(), 0.0)) /*/
      /*    v.size();*/
      /*std::cout << "Max block_size: " << minmax.second << "\n";*/
      /*std::cout << "Mean block_size: " << mean << "\nPartitioning time:" <<
       * tt*/
      /*          << std::endl;*/
      problem.partition(tolerance, options[j].data());
      problem.reorderToPartition();
      problem.renumberPoints();
      problem.loopGPUHierarchical(ITER_NUM);
    }
  }
  return 0;
}
