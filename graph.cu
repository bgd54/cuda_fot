#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "colouring.hpp"
#include "helper_cuda.h"
#include "problem.hpp"

constexpr std::size_t BLOCK_SIZE = 128;

/* Problem {{{1 */

__global__ void problem_stepGPU(float *point_weights, float *edge_weights,
                                std::size_t *edge_list, std::size_t *edge_inds,
                                float *out, std::size_t edge_num) {
  std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < edge_num) {
    std::size_t edge_ind = edge_inds[id];
    out[edge_list[2 * edge_ind + 1]] +=
        edge_weights[edge_ind] * point_weights[edge_list[2 * edge_ind]];
  }
}

__global__ void problem_stepGPUHierarchical(
    std::size_t *edge_list, float *point_weights, float *point_weights_out,
    float *edge_weights, std::size_t **points_to_be_cached,
    std::size_t cache_size, std::size_t *num_cached_points,
    std::uint8_t *edge_colours, std::uint8_t *num_edge_colours,
    std::size_t num_threads) {
  std::size_t bid = blockIdx.x;
  std::size_t thread_ind = bid * blockDim.x + threadIdx.x;
  std::size_t tid = threadIdx.x;

  extern __shared__ float shared[];
  float *point_cache_in = shared;
  float *point_cache_out = shared + cache_size;

  std::size_t in_ind = edge_list[2 * thread_ind];
  std::size_t out_ind = edge_list[2 * thread_ind + 1];

  std::uint8_t our_colour = edge_colours[thread_ind];
  if (thread_ind >= num_threads) {
    our_colour = num_edge_colours[bid];
  }

  std::size_t num_cached_point = num_cached_points[bid];

  // Cache in
  for (std::size_t i = 0; i < num_cached_point; i += blockDim.x) {
    if (i + tid < num_cached_point) {
      point_cache_in[i + tid] =
          point_weights[points_to_be_cached[bid][i + tid]];
      point_cache_out[i + tid] =
          point_weights_out[points_to_be_cached[bid][i + tid]];
    }
  }

  __syncthreads(); // TODO really syncthreads?

  // Computation
  float result = point_cache_in[in_ind] * edge_weights[thread_ind];

  for (std::size_t i = 0; i < num_edge_colours[bid]; ++i) {
    if (our_colour == i) {
      point_cache_out[out_ind] += result;
    }
    __syncthreads();
  }

  // Cache out
  for (std::size_t i = 0; i < num_cached_point; i += blockDim.x) {
    if (i + tid < num_cached_point) {
      point_weights_out[points_to_be_cached[bid][i + tid]] =
          point_cache_out[i + tid];
    }
  }
}

/* loopGPUEdgeCentred {{{2 */
void Problem::loopGPUEdgeCentred(std::size_t num, std::size_t reset_every) {
  std::vector<std::vector<std::size_t>> partition = graph.colourEdges();
  std::size_t num_of_colours = partition.size();
  std::size_t max_thread_num =
      std::max_element(
          partition.begin(), partition.end(),
          [](const std::vector<std::size_t> &a,
             const std::vector<std::size_t> &b) { return a.size() < b.size(); })
          ->size();
  std::size_t num_blocks = static_cast<std::size_t>(
      std::ceil(double(max_thread_num) / static_cast<double>(BLOCK_SIZE)));
  float *d_weights1, *d_weights2, *d_edge_weights;
  std::size_t *d_edge_list;
  std::vector<std::size_t *> d_partition;
  for (const std::vector<std::size_t> &colour : partition) {
    std::size_t *d_colour;
    std::size_t mem_size = sizeof(std::size_t) * colour.size();
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
                             sizeof(std::size_t) * 2 * graph.numEdges()));
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
                             sizeof(std::size_t) * 2 * graph.numEdges(),
                             cudaMemcpyHostToDevice));
  // Timer t;
  TIMER_START(t);
  for (std::size_t i = 0; i < num; ++i) {
    for (std::size_t c = 0; c < num_of_colours; ++c) {
      problem_stepGPU<<<num_blocks, BLOCK_SIZE>>>(
          d_weights1, d_edge_weights, d_edge_list, d_partition[c], d_weights2,
          partition[c].size());
      checkCudaErrors(cudaDeviceSynchronize());
    }
    checkCudaErrors(cudaMemcpy(d_weights1, d_weights2,
                               sizeof(float) * graph.numPoints(),
                               cudaMemcpyDeviceToDevice));
    TIMER_TOGGLE(t);
    if (reset_every && i % reset_every == reset_every - 1) {
      reset();
      checkCudaErrors(cudaMemcpy(d_weights1, point_weights,
                                 sizeof(float) * graph.numPoints(),
                                 cudaMemcpyHostToDevice));
    }
    TIMER_TOGGLE(t);
  }
  // long long time = t.getTime();
  TIMER_PRINT(t, "loopGPUEdgeCentred");
  checkCudaErrors(cudaMemcpy(point_weights, d_weights1,
                             sizeof(float) * graph.numPoints(),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_edge_list));
  checkCudaErrors(cudaFree(d_edge_weights));
  checkCudaErrors(cudaFree(d_weights2));
  checkCudaErrors(cudaFree(d_weights1));
  for (std::size_t i = 0; i < num_of_colours; ++i) {
    checkCudaErrors(cudaFree(d_partition[i]));
  }
  // return time;
}
/* 2}}} */

void Problem::loopGPUHierarchical(std::size_t num, std::size_t reset_every) {
  HierarchicalColourMemory memory(BLOCK_SIZE, *this);
  std::vector<float *> d_edge_weights;
  std::vector<std::vector<std::size_t *>> d_points_to_be_cached;
  std::vector<std::size_t *> d_edge_list;
  std::vector<std::uint8_t *> d_edge_colours;
  std::vector<std::uint8_t *> d_num_edge_colours;
  std::vector<std::size_t> shared_sizes;
  float *d_point_weights, *d_point_weights_out;
  std::vector<std::size_t *> d_num_cached_points;
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
  for (const HierarchicalColourMemory::MemoryOfOneColour &memory_of_one_colour :
       memory.colours) {
    float *d_fptr;
    std::size_t *d_sptr;
    std::uint8_t *d_uptr;
    checkCudaErrors(
        cudaMalloc((void **)&d_fptr,
                   sizeof(float) * memory_of_one_colour.edge_weights.size()));
    d_edge_weights.push_back(d_fptr);
    checkCudaErrors(
        cudaMemcpy(d_fptr, memory_of_one_colour.edge_weights.data(),
                   sizeof(float) * memory_of_one_colour.edge_weights.size(),
                   cudaMemcpyHostToDevice));
    d_points_to_be_cached.emplace_back();
    std::size_t shared_size = 0;
    std::vector<std::size_t> num_cached_points;
    for (const std::vector<std::size_t> &block_points_to_be_cached :
         memory_of_one_colour.points_to_be_cached) {
      checkCudaErrors(
          cudaMalloc((void **)&d_sptr,
                     sizeof(std::size_t) * block_points_to_be_cached.size()));
      d_points_to_be_cached.back().push_back(d_sptr);
      checkCudaErrors(
          cudaMemcpy(d_sptr, block_points_to_be_cached.data(),
                     sizeof(std::size_t) * block_points_to_be_cached.size(),
                     cudaMemcpyHostToDevice));
      shared_size = std::max(shared_size, block_points_to_be_cached.size());
      num_cached_points.push_back(block_points_to_be_cached.size());
    }
    assert(memory_of_one_colour.num_edge_colours.size() ==
           num_cached_points.size());
    checkCudaErrors(cudaMalloc((void **)&d_sptr,
                               sizeof(std::size_t) * num_cached_points.size()));
    d_num_cached_points.push_back(d_sptr);
    checkCudaErrors(cudaMemcpy(d_num_cached_points.back(),
                               num_cached_points.data(),
                               sizeof(std::size_t) * num_cached_points.size(),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&d_sptr,
                               sizeof(std::size_t) *
                                   memory_of_one_colour.edge_list.size()));
    d_edge_list.push_back(d_sptr);
    checkCudaErrors(
        cudaMemcpy(d_sptr, memory_of_one_colour.edge_list.data(),
                   sizeof(std::size_t) * memory_of_one_colour.edge_list.size(),
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
  for (std::size_t iteration = 0; iteration < num; ++iteration) {
    for (std::size_t colour_ind = 0; colour_ind < memory.colours.size();
         ++colour_ind) {
      std::size_t num_threads = memory.colours[colour_ind].edge_list.size();
      std::size_t num_blocks = static_cast<std::size_t>(
          std::ceil(static_cast<double>(num_threads) / BLOCK_SIZE));
      assert(num_blocks == memory.colours[colour_ind].num_edge_colours.size());
      std::size_t cache_size = sizeof(float) * 2 * shared_sizes[colour_ind];
      problem_stepGPUHierarchical<<<num_blocks, BLOCK_SIZE, cache_size>>>(
          d_edge_list[colour_ind], d_point_weights, d_point_weights_out,
          d_edge_weights[colour_ind], d_points_to_be_cached[colour_ind].data(),
          cache_size, d_num_cached_points[colour_ind],
          d_edge_colours[colour_ind], d_num_edge_colours[colour_ind],
          num_threads);
    }
    checkCudaErrors(cudaMemcpy(d_point_weights, d_point_weights_out,
                               sizeof(float) * graph.numPoints(),
                               cudaMemcpyDeviceToDevice));
    if (reset_every && iteration % reset_every == reset_every - 1) {
      reset();
      checkCudaErrors(cudaMemcpy(d_point_weights, point_weights,
                                 sizeof(float) * graph.numPoints(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(d_point_weights_out, point_weights,
                                 sizeof(float) * graph.numPoints(),
                                 cudaMemcpyHostToDevice));
    }
  }
  // ---------------
  // -  Finish up  -
  // ---------------
  checkCudaErrors(cudaMemcpy(point_weights, d_point_weights_out,
                             sizeof(float) * graph.numPoints(),
                             cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < memory.colours.size(); ++i) {
    checkCudaErrors(cudaFree(d_num_edge_colours[i]));
    checkCudaErrors(cudaFree(d_edge_colours[i]));
    checkCudaErrors(cudaFree(d_edge_list[i]));
    checkCudaErrors(cudaFree(d_num_cached_points[i]));
    for (std::size_t j = 0; j < d_points_to_be_cached[i].size(); ++j) {
      checkCudaErrors(cudaFree(d_points_to_be_cached[i][j]));
    }
    checkCudaErrors(cudaFree(d_edge_weights[i]));
  }
  checkCudaErrors(cudaFree(d_point_weights_out));
  checkCudaErrors(cudaFree(d_point_weights));
}

/* 1}}} */

/* tests {{{1 */
void testTwoCPUImplementations(std::size_t num) {
  // std::cout.precision(3);
  std::cout << "CPU point vs CPU edge" << std::endl;
  std::vector<float> result1;
  std::size_t N = 1000;
  std::size_t M = 2000;
  std::size_t reset_every = 10;
  {
    srand(1);
    Problem problem(N, M);
    problem.loopCPUPointCentred(num, reset_every);
    float mx = 0;
    for (std::size_t i = 0; i < problem.graph.numPoints(); ++i) {
      // std::cout << problem.point_weights[i] << " ";
      result1.push_back(problem.point_weights[i]);
      mx = std::max(mx, std::abs(problem.point_weights[i]));
    }
    std::cerr << "max: " << mx << std::endl;
    // std::cout << std::endl;
    // for (std::size_t i = 0; i < problem.graph.numEdges(); ++i) {
    //  std::cout << problem.edge_weights[i] << " ";
    //}
    // std::cout << std::endl;
  }

  double rms = 0;
  {
    srand(1);
    Problem problem(N, M);
    problem.loopCPUEdgeCentred(num, reset_every);
    // for (std::size_t i = 0; i < problem.graph.numEdges(); ++i) {
    //  std::cout << problem.edge_weights[i] << " ";
    //}
    // std::cout << std::endl;
    for (std::size_t i = 0; i < problem.graph.numPoints(); ++i) {
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
  //  for (std::size_t a : vv) {
  //    std::cout << a << " ";
  //  }
  //  std::cout << std::endl;
  //}
  std::cout << v.size() << " " << v.at(0).size() << " " << v.at(1).size()
            << std::endl;
}

void testGPUSolution(std::size_t num) {
  std::cout << "CPU edge vs GPU edge" << std::endl;

  std::vector<float> result1;
  double rms = 0;
  std::size_t N = 1000;
  std::size_t M = 2000;
  std::size_t reset_every = 10;
  {
    srand(1);
    Problem problem(N, M);
    // std::cout << "CPU time: ";
    // long long time = problem.loopCPUEdgeCentred(num);
    problem.loopCPUEdgeCentred(num, reset_every);
    // std::cout << time << " ms" << std::endl;
    for (std::size_t i = 0; i < problem.graph.numPoints(); ++i) {
      result1.push_back(problem.point_weights[i]);
    }
  }

  {
    srand(1);
    Problem problem(N, M);
    // long long time = problem.loopGPUEdgeCentred(num);
    problem.loopGPUEdgeCentred(num);
    // std::cout << "GPU time " << time << " ms" << std::endl;
    for (std::size_t i = 0; i < problem.graph.numPoints(); ++i) {
      rms += std::pow(problem.point_weights[i] - result1[i], 2);
    }
    rms = std::pow(rms / result1.size(), 0.5);
    std::cout << "RMS: " << rms << std::endl;
  }

  cudaDeviceReset();
}

void testHierarchicalColouring() {
  Problem problem(1000, 2000);
  constexpr std::size_t BLOCK_SIZE = 128;
  constexpr bool PRINT_RESULT = false;
  if (PRINT_RESULT) {
    std::cout << "Edge weights: ";
    for (std::size_t i = 0; i < problem.graph.numEdges(); ++i) {
      std::cout << problem.edge_weights[i] << " ";
    }
    std::cout << "Edge list: ";
    for (std::size_t i = 0; i < problem.graph.numEdges(); ++i) {
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
      for (const auto &v : c.points_to_be_cached) {
        std::cout << std::endl;
        for (std::size_t p : v) {
          std::cout << p << " ";
        }
      }
      std::cout << std::endl << "Edge list: ";
      for (std::size_t i = 0; i < c.edge_list.size(); i += 2) {
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

void testGPUHierarchicalSolution(std::size_t num) {
  std::cout << "CPU edge vs GPU hierarchical" << std::endl;

  std::vector<float> result1;
  double rms = 0;
  std::size_t N = 1000;
  std::size_t M = 2000;
  std::size_t reset_every = 10;
  {
    srand(1);
    Problem problem(N, M);
    // std::cout << "CPU time: ";
    // long long time = problem.loopCPUEdgeCentred(num);
    problem.loopCPUEdgeCentred(num, reset_every);
    // std::cout << time << " ms" << std::endl;
    for (std::size_t i = 0; i < problem.graph.numPoints(); ++i) {
      result1.push_back(problem.point_weights[i]);
    }
  }

  {
    srand(1);
    Problem problem(N, M);
    // long long time = problem.loopGPUEdgeCentred(num);
    problem.loopGPUHierarchical(num);
    // std::cout << "GPU time " << time << " ms" << std::endl;
    for (std::size_t i = 0; i < problem.graph.numPoints(); ++i) {
      rms += std::pow(problem.point_weights[i] - result1[i], 2);
    }
    rms = std::pow(rms / result1.size(), 0.5);
    std::cout << "RMS: " << rms << std::endl;
  }

  cudaDeviceReset();
}
/* 1}}} */

int main(int argc, const char **argv) {
  findCudaDevice(argc, argv);
  // testTwoCPUImplementations(99);
  testGPUSolution(100);
  // testHierarchicalColouring();
}

// vim:set et sw=2 ts=2 fdm=marker:
