#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

//#define MY_SIZE int
using MY_SIZE = std::uint32_t;

#include "colouring.hpp"
#include "helper_cuda.h"
#include "problem.hpp"

constexpr MY_SIZE BLOCK_SIZE = 128;

/* problem_stepGPU {{{1 */
__global__ void problem_stepGPU(float *point_weights, float *edge_weights,
                                MY_SIZE *edge_list, MY_SIZE *edge_inds,
                                float *out, MY_SIZE edge_num) {
  MY_SIZE id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < edge_num) {
    MY_SIZE edge_ind = edge_inds[id];
    out[edge_list[2 * edge_ind + 1]] +=
        edge_weights[edge_ind] * point_weights[edge_list[2 * edge_ind]];
  }
}
/* 1}}} */

/* problem_stepGPUHierarchical {{{1 */
__global__ void problem_stepGPUHierarchical(
    MY_SIZE *edge_list, float *point_weights, float *point_weights_out,
    float *edge_weights, MY_SIZE *points_to_be_cached, MY_SIZE cache_size,
    MY_SIZE *points_to_be_cached_offsets, std::uint8_t *edge_colours,
    std::uint8_t *num_edge_colours, MY_SIZE num_threads) {
  MY_SIZE bid = blockIdx.x;
  MY_SIZE thread_ind = bid * blockDim.x + threadIdx.x;
  MY_SIZE tid = threadIdx.x;

  extern __shared__ float shared[];
  float *point_cache_in = shared;
  float *point_cache_out = shared + cache_size;

  MY_SIZE out_ind, in_ind;

  std::uint8_t our_colour;
  if (thread_ind >= num_threads) {
    our_colour = num_edge_colours[bid];
  } else {
    our_colour = edge_colours[thread_ind];
  }

  MY_SIZE points_offset = points_to_be_cached_offsets[bid];
  MY_SIZE num_cached_point =
      points_to_be_cached_offsets[bid + 1] - points_offset;
  if (num_cached_point > cache_size) {
    printf("ERROR: Bid: %d Tid: %d, num_cached_point: %d cache_size: %d\n", bid,
           tid, num_cached_point, cache_size);
  }
  // printf("DEBUG: bid: %d tid: %d num_cached_point: %d cache_size:
  // %d\n",bid,tid,num_cached_point,cache_size);
  // printf("       thread_ind: %d, num_threads: %d\n",thread_ind,num_threads);

  // Cache in
  for (MY_SIZE i = 0; i < num_cached_point; i += blockDim.x) {
    if (i + tid < num_cached_point) {
      point_cache_in[i + tid] =
          point_weights[points_to_be_cached[points_offset + i + tid]];
      point_cache_out[i + tid] =
          point_weights_out[points_to_be_cached[points_offset + i + tid]];
    }
  }

  __syncthreads(); // TODO really syncthreads?

  // Computation
  float result = 0;
  if (thread_ind < num_threads) {
    in_ind = edge_list[2 * thread_ind];
    out_ind = edge_list[2 * thread_ind + 1];
    result = point_cache_in[in_ind] * edge_weights[thread_ind];
  }

  for (MY_SIZE i = 0; i < num_edge_colours[bid]; ++i) {
    if (our_colour == i) {
      point_cache_out[out_ind] += result;
    }
    __syncthreads();
  }

  // Cache out
  for (MY_SIZE i = 0; i < num_cached_point; i += blockDim.x) {
    if (i + tid < num_cached_point) {
      point_weights_out[points_to_be_cached[points_offset + i + tid]] =
          point_cache_out[i + tid];
    }
  }
}
/* 1}}} */

/* loopGPUEdgeCentred {{{1 */
void Problem::loopGPUEdgeCentred(MY_SIZE num, MY_SIZE reset_every) {
  std::vector<std::vector<MY_SIZE>> partition = graph.colourEdges();
  MY_SIZE num_of_colours = partition.size();
  MY_SIZE max_thread_num = std::max_element(partition.begin(), partition.end(),
                                            [](const std::vector<MY_SIZE> &a,
                                               const std::vector<MY_SIZE> &b) {
                                              return a.size() < b.size();
                                            })
                               ->size();
  MY_SIZE num_blocks = static_cast<MY_SIZE>(
      std::ceil(double(max_thread_num) / static_cast<double>(BLOCK_SIZE)));
  float *d_weights1, *d_weights2, *d_edge_weights;
  MY_SIZE *d_edge_list;
  std::vector<MY_SIZE *> d_partition;
  for (const std::vector<MY_SIZE> &colour : partition) {
    MY_SIZE *d_colour;
    MY_SIZE mem_size = sizeof(MY_SIZE) * colour.size();
    checkCudaErrors(cudaMalloc((void **)&d_colour, mem_size));
    checkCudaErrors(
        cudaMemcpy(d_colour, colour.data(), mem_size, cudaMemcpyHostToDevice));
    d_partition.push_back(d_colour);
  }
  checkCudaErrors(
      cudaMalloc((void **)&d_weights1, sizeof(float) * graph.numPoints()));
  checkCudaErrors(
      cudaMalloc((void **)&d_weights2, sizeof(float) * graph.numPoints()));
  checkCudaErrors(
      cudaMalloc((void **)&d_edge_weights, sizeof(float) * graph.numEdges()));
  checkCudaErrors(cudaMalloc((void **)&d_edge_list,
                             sizeof(MY_SIZE) * 2 * graph.numEdges()));
  checkCudaErrors(cudaMemcpy(d_weights1, point_weights,
                             sizeof(float) * graph.numPoints(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_weights2, point_weights,
                             sizeof(float) * graph.numPoints(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_edge_weights, edge_weights,
                             sizeof(float) * graph.numEdges(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_edge_list, graph.edge_list,
                             sizeof(MY_SIZE) * 2 * graph.numEdges(),
                             cudaMemcpyHostToDevice));
  // Timer t;
  TIMER_START(t);
  for (MY_SIZE i = 0; i < num; ++i) {
    for (MY_SIZE c = 0; c < num_of_colours; ++c) {
      problem_stepGPU<<<num_blocks, BLOCK_SIZE>>>(
          d_weights1, d_edge_weights, d_edge_list, d_partition[c], d_weights2,
          partition[c].size());
      checkCudaErrors(cudaDeviceSynchronize());
    }
    TIMER_TOGGLE(t);
    if (reset_every && i % reset_every == reset_every - 1) {
      reset();
      // Copy to d_weights2 that is currently holding the result, the next
      // copy will put it into d_weights1 also.
      checkCudaErrors(cudaMemcpy(d_weights2, point_weights,
                                 sizeof(float) * graph.numPoints(),
                                 cudaMemcpyHostToDevice));
    }
    TIMER_TOGGLE(t);
    checkCudaErrors(cudaMemcpy(d_weights1, d_weights2,
                               sizeof(float) * graph.numPoints(),
                               cudaMemcpyDeviceToDevice));
  }
  PRINT_BANDWIDTH(t, "loopGPUEdgeCentred",
                  sizeof(float) * (2 * graph.numPoints() + graph.numEdges()) *
                      num,
                  (sizeof(float) * graph.numPoints() * 2 +  // d_weights
                   sizeof(float) * graph.numEdges() +       // d_edge_weights
                   sizeof(MY_SIZE) * graph.numEdges() * 2 + // d_edge_list
                   sizeof(MY_SIZE) * graph.numEdges() * 2   // d_partition
                   ) * num);
  checkCudaErrors(cudaMemcpy(point_weights, d_weights1,
                             sizeof(float) * graph.numPoints(),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_edge_list));
  checkCudaErrors(cudaFree(d_edge_weights));
  checkCudaErrors(cudaFree(d_weights2));
  checkCudaErrors(cudaFree(d_weights1));
  for (MY_SIZE i = 0; i < num_of_colours; ++i) {
    checkCudaErrors(cudaFree(d_partition[i]));
  }
}
/* 1}}} */

/* loopGPUHierarchical {{{1 */
void Problem::loopGPUHierarchical(MY_SIZE num, MY_SIZE reset_every) {
  HierarchicalColourMemory memory(BLOCK_SIZE, *this);
  std::vector<float *> d_edge_weights;
  std::vector<MY_SIZE *> d_points_to_be_cached;
  std::vector<MY_SIZE *> d_edge_list;
  std::vector<std::uint8_t *> d_edge_colours;
  std::vector<std::uint8_t *> d_num_edge_colours;
  std::vector<MY_SIZE> shared_sizes;
  float *d_point_weights, *d_point_weights_out;
  std::vector<MY_SIZE *> d_points_to_be_cached_offsets;
  checkCudaErrors(
      cudaMalloc((void **)&d_point_weights, sizeof(float) * graph.numPoints()));
  checkCudaErrors(cudaMalloc((void **)&d_point_weights_out,
                             sizeof(float) * graph.numPoints()));
  checkCudaErrors(cudaMemcpy(d_point_weights, point_weights,
                             sizeof(float) * graph.numPoints(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_point_weights_out, point_weights,
                             sizeof(float) * graph.numPoints(),
                             cudaMemcpyHostToDevice));
  MY_SIZE total_cache_size = 0; // for bandwidth calculations
  for (const HierarchicalColourMemory::MemoryOfOneColour &memory_of_one_colour :
       memory.colours) {
    float *d_fptr;
    MY_SIZE *d_sptr;
    std::uint8_t *d_uptr;
    checkCudaErrors(
        cudaMalloc((void **)&d_fptr,
                   sizeof(float) * memory_of_one_colour.edge_weights.size()));
    d_edge_weights.push_back(d_fptr);
    checkCudaErrors(
        cudaMemcpy(d_fptr, memory_of_one_colour.edge_weights.data(),
                   sizeof(float) * memory_of_one_colour.edge_weights.size(),
                   cudaMemcpyHostToDevice));
    MY_SIZE shared_size = 0;
    checkCudaErrors(cudaMalloc(
        (void **)&d_sptr,
        sizeof(MY_SIZE) * memory_of_one_colour.points_to_be_cached.size()));
    d_points_to_be_cached.push_back(d_sptr);
    checkCudaErrors(cudaMemcpy(
        d_sptr, memory_of_one_colour.points_to_be_cached.data(),
        sizeof(MY_SIZE) * memory_of_one_colour.points_to_be_cached.size(),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(
        (void **)&d_sptr,
        sizeof(MY_SIZE) *
            memory_of_one_colour.points_to_be_cached_offsets.size()));
    d_points_to_be_cached_offsets.push_back(d_sptr);
    checkCudaErrors(cudaMemcpy(
        d_sptr, memory_of_one_colour.points_to_be_cached_offsets.data(),
        sizeof(MY_SIZE) *
            memory_of_one_colour.points_to_be_cached_offsets.size(),
        cudaMemcpyHostToDevice));
    for (MY_SIZE i = 1;
         i < memory_of_one_colour.points_to_be_cached_offsets.size(); ++i) {
      shared_size = std::max<MY_SIZE>(
          shared_size,
          memory_of_one_colour.points_to_be_cached_offsets[i] -
              memory_of_one_colour.points_to_be_cached_offsets[i - 1]);
    }
    shared_sizes.push_back(shared_size);
    total_cache_size += memory_of_one_colour.points_to_be_cached.size();
    checkCudaErrors(
        cudaMalloc((void **)&d_sptr,
                   sizeof(MY_SIZE) * memory_of_one_colour.edge_list.size()));
    d_edge_list.push_back(d_sptr);
    checkCudaErrors(
        cudaMemcpy(d_sptr, memory_of_one_colour.edge_list.data(),
                   sizeof(MY_SIZE) * memory_of_one_colour.edge_list.size(),
                   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&d_uptr,
                               sizeof(std::uint8_t) *
                                   memory_of_one_colour.edge_colours.size()));
    d_edge_colours.push_back(d_uptr);
    checkCudaErrors(cudaMemcpy(d_uptr, memory_of_one_colour.edge_colours.data(),
                               sizeof(std::uint8_t) *
                                   memory_of_one_colour.edge_colours.size(),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(
        (void **)&d_uptr,
        sizeof(std::uint8_t) * memory_of_one_colour.num_edge_colours.size()));
    d_num_edge_colours.push_back(d_uptr);
    checkCudaErrors(cudaMemcpy(
        d_uptr, memory_of_one_colour.num_edge_colours.data(),
        sizeof(std::uint8_t) * memory_of_one_colour.num_edge_colours.size(),
        cudaMemcpyHostToDevice));
  }
  // -----------------------
  // -  Start computation  -
  // -----------------------
  TIMER_START(t);
  MY_SIZE total_num_blocks = 0; // for bandwidth calculations
  for (MY_SIZE iteration = 0; iteration < num; ++iteration) {
    for (MY_SIZE colour_ind = 0; colour_ind < memory.colours.size();
         ++colour_ind) {
      assert(memory.colours[colour_ind].edge_list.size() % 2 == 0);
      MY_SIZE num_threads = memory.colours[colour_ind].edge_list.size() / 2;
      MY_SIZE num_blocks = static_cast<MY_SIZE>(
          std::ceil(static_cast<double>(num_threads) / BLOCK_SIZE));
      assert(num_blocks == memory.colours[colour_ind].num_edge_colours.size());
      MY_SIZE cache_size = sizeof(float) * 2 * shared_sizes[colour_ind];
      problem_stepGPUHierarchical<<<num_blocks, BLOCK_SIZE, cache_size>>>(
          d_edge_list[colour_ind], d_point_weights, d_point_weights_out,
          d_edge_weights[colour_ind], d_points_to_be_cached[colour_ind],
          shared_sizes[colour_ind], d_points_to_be_cached_offsets[colour_ind],
          d_edge_colours[colour_ind], d_num_edge_colours[colour_ind],
          num_threads);
      checkCudaErrors(cudaDeviceSynchronize());
      total_num_blocks += num_blocks;
    }
    checkCudaErrors(cudaMemcpy(d_point_weights, d_point_weights_out,
                               sizeof(float) * graph.numPoints(),
                               cudaMemcpyDeviceToDevice));
    if (reset_every && iteration % reset_every == reset_every - 1) {
      TIMER_TOGGLE(t);
      reset();
      checkCudaErrors(cudaMemcpy(d_point_weights, point_weights,
                                 sizeof(float) * graph.numPoints(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(d_point_weights_out, point_weights,
                                 sizeof(float) * graph.numPoints(),
                                 cudaMemcpyHostToDevice));
      TIMER_TOGGLE(t);
    }
  }
  PRINT_BANDWIDTH(
      t, "GPU HierarchicalColouring",
      num * (2 * graph.numPoints() + graph.numEdges()) * sizeof(float),
      num * (sizeof(float) * graph.numPoints() * 2 +  // point_weights
             sizeof(float) * graph.numEdges() +       // edge_weights
             sizeof(MY_SIZE) * graph.numEdges() * 2 + // edge_list
             sizeof(MY_SIZE) * total_cache_size +
             sizeof(MY_SIZE) *
                 (total_num_blocks +
                  memory.colours.size()) + // points_to_be_cached_offsets
             sizeof(std::uint8_t) * graph.numEdges() // edge_colours
             ));
  std::cout << "  recycling factor: "
            << 2 * static_cast<double>(total_cache_size) / (2 * graph.numEdges())
            << std::endl;
  // ---------------
  // -  Finish up  -
  // ---------------
  checkCudaErrors(cudaMemcpy(point_weights, d_point_weights,
                             sizeof(float) * graph.numPoints(),
                             cudaMemcpyDeviceToHost));
  for (MY_SIZE i = 0; i < memory.colours.size(); ++i) {
    checkCudaErrors(cudaFree(d_num_edge_colours[i]));
    checkCudaErrors(cudaFree(d_edge_colours[i]));
    checkCudaErrors(cudaFree(d_edge_list[i]));
    checkCudaErrors(cudaFree(d_points_to_be_cached_offsets[i]));
    checkCudaErrors(cudaFree(d_points_to_be_cached[i]));
    checkCudaErrors(cudaFree(d_edge_weights[i]));
  }
  checkCudaErrors(cudaFree(d_point_weights_out));
  checkCudaErrors(cudaFree(d_point_weights));
}
/* 1}}} */

/* tests {{{1 */
void testTwoCPUImplementations(MY_SIZE num) {
  // std::cout.precision(3);
  std::cout << "CPU point vs CPU edge" << std::endl;
  std::vector<float> result1;
  MY_SIZE N = 1000;
  MY_SIZE M = 2000;
  MY_SIZE reset_every = 10;
  {
    srand(1);
    Problem problem(N, M);
    problem.loopCPUPointCentred(num, reset_every);
    float mx = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      // std::cout << problem.point_weights[i] << " ";
      result1.push_back(problem.point_weights[i]);
      mx = std::max(mx, std::abs(problem.point_weights[i]));
    }
    std::cerr << "max: " << mx << std::endl;
    // std::cout << std::endl;
    // for (MY_SIZE i = 0; i < problem.graph.numEdges(); ++i) {
    //  std::cout << problem.edge_weights[i] << " ";
    //}
    // std::cout << std::endl;
  }

  double rms = 0;
  {
    srand(1);
    Problem problem(N, M);
    problem.loopCPUEdgeCentred(num, reset_every);
    // for (MY_SIZE i = 0; i < problem.graph.numEdges(); ++i) {
    //  std::cout << problem.edge_weights[i] << " ";
    //}
    // std::cout << std::endl;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      // std::cout << problem.point_weights[i] << " ";
      rms += std::pow(problem.point_weights[i] - result1[i], 2);
    }
    // std::cout << std::endl;
    rms = std::pow(rms / result1.size(), 0.5);
    std::cout << "RMS: " << rms << std::endl;
  }
}

void testColours() {
  Graph graph(1000, 2000);
  auto v = graph.colourEdges();

  // for (const auto &vv : v) {
  //  for (MY_SIZE a : vv) {
  //    std::cout << a << " ";
  //  }
  //  std::cout << std::endl;
  //}
  std::cout << v.size() << " " << v.at(0).size() << " " << v.at(1).size()
            << std::endl;
}

void testGPUSolution(MY_SIZE num) {
  std::cout << "CPU edge vs GPU edge" << std::endl;

  std::vector<float> result1;
  double rms = 0;
  MY_SIZE N = 1000;
  MY_SIZE M = 2000;
  MY_SIZE reset_every = 10;
  {
    srand(1);
    Problem problem(N, M);
    problem.loopCPUEdgeCentred(num, reset_every);
    float abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      result1.push_back(problem.point_weights[i]);
      abs_max = std::max(abs_max, problem.point_weights[i]);
    }
    std::cout << "Abs max: " << abs_max << std::endl;
  }

  {
    srand(1);
    Problem problem(N, M);
    problem.loopGPUEdgeCentred(num, reset_every);
    float abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      rms += std::pow(problem.point_weights[i] - result1[i], 2);
      abs_max = std::max(abs_max, problem.point_weights[i]);
    }
    rms = std::pow(rms / result1.size(), 0.5);
    std::cout << "Abs max: " << abs_max << std::endl;
    std::cout << "RMS: " << rms << std::endl;
  }
}

void testHierarchicalColouring() {
  Problem problem(2, 4);
  constexpr MY_SIZE BLOCK_SIZE = 3;
  constexpr bool PRINT_RESULT = true;
  if (PRINT_RESULT) {
    std::cout << "Edge weights: ";
    for (MY_SIZE i = 0; i < problem.graph.numEdges(); ++i) {
      std::cout << problem.edge_weights[i] << " ";
    }
    std::cout << std::endl << "Edge list: ";
    for (MY_SIZE i = 0; i < problem.graph.numEdges(); ++i) {
      std::cout << problem.graph.edge_list[2 * i] << "->"
                << problem.graph.edge_list[2 * i + 1] << std::endl;
    }
    std::cout << std::endl;
  }
  Timer t;
  HierarchicalColourMemory memory(BLOCK_SIZE, problem);
  std::cout << "memory colouring time: ";
  t.printTime();
  std::cout << std::endl;
  if (PRINT_RESULT) {
    for (const auto &c : memory.colours) {
      std::cout << "================================================"
                << std::endl
                << "Memory:" << std::endl;
      std::cout << "Edge weights: ";
      for (float w : c.edge_weights) {
        std::cout << w << " ";
      }
      std::cout << std::endl << "Points to be cached: ";
      for (const auto &p : c.points_to_be_cached) {
        std::cout << p << " ";
      }
      std::cout << std::endl << "Points to be cached (offsets): ";
      for (const auto &p : c.points_to_be_cached_offsets) {
        std::cout << p << " ";
      }
      std::cout << std::endl << "Edge list: ";
      for (MY_SIZE i = 0; i < c.edge_list.size(); i += 2) {
        std::cout << c.edge_list[i] << "->" << c.edge_list[i + 1] << " ";
      }
      std::cout << std::endl;
      std::cout << "Num of edge colours: ";
      for (std::uint8_t nec : c.num_edge_colours) {
        std::cout << static_cast<unsigned>(nec) << " ";
      }
      std::cout << std::endl;
      std::cout << "Edge colours: ";
      for (std::uint8_t cc : c.edge_colours) {
        std::cout << static_cast<int>(cc) << " ";
      }
      std::cout << std::endl;
    }
  }
}

void testGPUHierarchicalSolution(MY_SIZE num) {
  std::cout << "CPU edge vs GPU hierarchical" << std::endl;

  std::vector<float> result1;
  double rms = 0;
  MY_SIZE N = 1000;
  MY_SIZE M = 2000;
  MY_SIZE reset_every = 10;
  {
    srand(1);
    Problem problem(N, M);
    problem.loopCPUEdgeCentred(num, reset_every);
    float abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      result1.push_back(problem.point_weights[i]);
      abs_max = std::max(abs_max, problem.point_weights[i]);
    }
    std::cout << "Abs max: " << abs_max << std::endl;
  }

  {
    srand(1);
    Problem problem(N, M);
    problem.loopGPUHierarchical(num, reset_every);
    float abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      rms += std::pow(problem.point_weights[i] - result1[i], 2);
      abs_max = std::max(abs_max, problem.point_weights[i]);
    }
    std::cout << "Abs max: " << abs_max << std::endl;
    rms = std::pow(rms / result1.size(), 0.5);
    std::cout << "RMS: " << rms << std::endl;
  }
}

using implementation_algorithm_t = void (Problem::*)(MY_SIZE, MY_SIZE);
void testTwoImplementations(MY_SIZE num, MY_SIZE N, MY_SIZE M,
                            MY_SIZE reset_every,
                            implementation_algorithm_t algorithm1,
                            implementation_algorithm_t algorithm2) {
  std::vector<float> result1;
  double rms = 0;
  {
    srand(1);
    Problem problem(N, M);
    (problem.*algorithm1)(num, reset_every);
    float abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      result1.push_back(problem.point_weights[i]);
      abs_max = std::max(abs_max, problem.point_weights[i]);
    }
    std::cout << "Abs max: " << abs_max << std::endl;
  }

  {
    srand(1);
    Problem problem(N, M);
    (problem.*algorithm2)(num, reset_every);
    float abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      rms += std::pow(problem.point_weights[i] - result1[i], 2);
      abs_max = std::max(abs_max, problem.point_weights[i]);
    }
    std::cout << "Abs max: " << abs_max << std::endl;
    rms = std::pow(rms / result1.size(), 0.5);
    std::cout << "RMS: " << rms << std::endl;
  }
}
/* 1}}} */

int main(int argc, const char **argv) {
  findCudaDevice(argc, argv);
  // testGPUSolution(std::atoi(argv[1]));
  // testHierarchicalColouring();
  // testGPUHierarchicalSolution(11);
  MY_SIZE num = 99;
  MY_SIZE N = 1000;
  MY_SIZE M = 2000;
  MY_SIZE reset_every = 10;
  std::cout << "GPU global edge vs GPU hierarchical edge" << std::endl;
  testTwoImplementations(num, N, M, reset_every, &Problem::loopGPUEdgeCentred,
                         &Problem::loopGPUHierarchical);
  cudaDeviceReset();
}

// vim:set et sw=2 ts=2 fdm=marker:
