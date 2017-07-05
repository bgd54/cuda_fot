#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

#include "colouring.hpp"
#include "helper_cuda.h"
#include "problem.hpp"

constexpr MY_SIZE BLOCK_SIZE = 128;

/* problem_stepGPU {{{1 */
template <unsigned Dim = 1, bool SOA = false, typename DataType>
__global__ void problem_stepGPU(const DataType *__restrict__ point_weights,
                                const DataType *__restrict__ edge_weights,
                                const MY_SIZE *__restrict__ edge_list,
                                const MY_SIZE *__restrict__ edge_inds,
                                DataType *__restrict__ out,
                                const MY_SIZE edge_num,
                                const MY_SIZE point_num) {
  MY_SIZE id = blockIdx.x * blockDim.x + threadIdx.x;
  DataType inc[2 * Dim];
  if (id < edge_num) {
    MY_SIZE edge_ind = edge_inds[id];
#pragma unroll
    for (MY_SIZE d = 0; d < Dim; ++d) {
      MY_SIZE ind_left, ind_right;
      if (SOA) {
        ind_left = d * point_num + edge_list[2 * edge_ind];
        ind_right = d * point_num + edge_list[2 * edge_ind + 1];
      } else {
        ind_left = edge_list[2 * edge_ind] * Dim + d;
        ind_right = edge_list[2 * edge_ind + 1] * Dim + d;
      }
      inc[d] =
          out[ind_right] + edge_weights[edge_ind] * point_weights[ind_left];
      inc[d + Dim] =
          out[ind_left] + edge_weights[edge_ind] * point_weights[ind_right];
    }
#pragma unroll
    for (MY_SIZE d = 0; d < Dim; ++d) {
      MY_SIZE ind_left, ind_right;
      if (SOA) {
        ind_left = d * point_num + edge_list[2 * edge_ind];
        ind_right = d * point_num + edge_list[2 * edge_ind + 1];
      } else {
        ind_left = edge_list[2 * edge_ind] * Dim + d;
        ind_right = edge_list[2 * edge_ind + 1] * Dim + d;
      }
      out[ind_right] = inc[d];
      out[ind_left] = inc[d + Dim];
    }
  }
}
/* 1}}} */

/* problem_stepGPUHierarchical {{{1 */
template <unsigned Dim = 1, bool SOA = false, typename DataType = float>
__global__ void problem_stepGPUHierarchical(
    const MY_SIZE *__restrict__ edge_list,
    const DataType *__restrict__ point_weights,
    DataType *__restrict__ point_weights_out,
    const DataType *__restrict__ edge_weights,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    const std::uint8_t *__restrict__ edge_colours,
    const std::uint8_t *__restrict__ num_edge_colours, MY_SIZE num_threads,
    const MY_SIZE num_points) {
  MY_SIZE bid = blockIdx.x;
  MY_SIZE thread_ind = bid * blockDim.x + threadIdx.x;
  MY_SIZE tid = threadIdx.x;

  MY_SIZE cache_points_offset = points_to_be_cached_offsets[bid];
  MY_SIZE num_cached_point =
      points_to_be_cached_offsets[bid + 1] - cache_points_offset;

  extern __shared__ __align__(alignof(DataType)) unsigned char shared[];
  DataType *point_cache = reinterpret_cast<DataType *>(shared);

  std::uint8_t our_colour;
  if (thread_ind >= num_threads) {
    our_colour = num_edge_colours[bid];
  } else {
    our_colour = edge_colours[thread_ind];
  }

  // Cache in
  for (MY_SIZE i = 0; i < num_cached_point; i += blockDim.x) {
    for (MY_SIZE d = 0; d < Dim; ++d) {
      MY_SIZE c_ind, g_ind;
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i + tid];
      if (i + tid < num_cached_point) {
        if (SOA) {
          g_ind = d * num_points + g_point_to_be_cached;
          c_ind = d * num_cached_point + (i + tid);
        } else {
          g_ind = g_point_to_be_cached * Dim + d;
          c_ind = (i + tid) * Dim + d;
        }
        point_cache[c_ind] = point_weights[g_ind];
      }
    }
  }

  __syncthreads();

  // Computation
  DataType increment[Dim * 2];
  if (thread_ind < num_threads) {
    for (MY_SIZE d = 0; d < Dim; ++d) {
      MY_SIZE left_ind, right_ind;
      if (SOA) {
        left_ind = d * num_cached_point + edge_list[2 * thread_ind];
        right_ind = d * num_cached_point + edge_list[2 * thread_ind + 1];
      } else {
        left_ind = edge_list[2 * thread_ind] * Dim + d;
        right_ind = edge_list[2 * thread_ind + 1] * Dim + d;
      }
      increment[d] = point_cache[left_ind] * edge_weights[thread_ind];
      increment[d + Dim] = point_cache[right_ind] * edge_weights[thread_ind];
    }
  }

  __syncthreads();

  // Clear cache
  for (MY_SIZE i = 0; i < num_cached_point; i += blockDim.x) {
    if (i + tid < num_cached_point) {
      for (MY_SIZE d = 0; d < Dim; ++d) {
        MY_SIZE c_ind;
        if (SOA) {
          c_ind = d * num_cached_point + (i + tid);
        } else {
          c_ind = (i + tid) * Dim + d;
        }
        point_cache[c_ind] = 0;
      }
    }
  }

  __syncthreads();

  // Accumulate increment
  for (MY_SIZE i = 0; i < num_edge_colours[bid]; ++i) {
    if (our_colour == i) {
      for (MY_SIZE d = 0; d < Dim; ++d) {
        MY_SIZE left_ind, right_ind;
        if (SOA) {
          left_ind = d * num_cached_point + edge_list[2 * thread_ind];
          right_ind = d * num_cached_point + edge_list[2 * thread_ind + 1];
        } else {
          left_ind = edge_list[2 * thread_ind] * Dim + d;
          right_ind = edge_list[2 * thread_ind + 1] * Dim + d;
        }
        point_cache[right_ind] += increment[d];
        point_cache[left_ind] += increment[d + Dim];
      }
    }
    __syncthreads();
  }

  // Cache out
  for (MY_SIZE i = 0; i < num_cached_point; i += blockDim.x) {
    if (i + tid < num_cached_point) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i + tid];
      DataType result[Dim];
#pragma unroll
      for (MY_SIZE d = 0; d < Dim; ++d) {
        MY_SIZE write_c_ind, write_g_ind;
        if (SOA) {
          write_g_ind = d * num_points + g_point_to_be_cached;
          write_c_ind = d * num_cached_point + (i + tid);
        } else {
          write_g_ind = g_point_to_be_cached * Dim + d;
          write_c_ind = (i + tid) * Dim + d;
        }
        result[d] = point_weights_out[write_g_ind] + point_cache[write_c_ind];
      }
#pragma unroll
      for (MY_SIZE d = 0; d < Dim; ++d) {
        MY_SIZE write_g_ind;
        if (SOA) {
          write_g_ind = d * num_points + g_point_to_be_cached;
        } else {
          write_g_ind = g_point_to_be_cached * Dim + d;
        }
        point_weights_out[write_g_ind] = result[d];
      }
    }
  }
}
/* 1}}} */

/* loopGPUEdgeCentred {{{1 */
template <unsigned Dim, bool SOA, typename DataType>
void Problem<Dim, SOA, DataType>::loopGPUEdgeCentred(MY_SIZE num,
                                                     MY_SIZE reset_every) {
  std::vector<std::vector<MY_SIZE>> partition = graph.colourEdges();
  MY_SIZE num_of_colours = partition.size();
  assert(num_of_colours > 0);
  MY_SIZE max_thread_num = std::max_element(partition.begin(), partition.end(),
                                            [](const std::vector<MY_SIZE> &a,
                                               const std::vector<MY_SIZE> &b) {
                                              return a.size() < b.size();
                                            })
                               ->size();
  MY_SIZE num_blocks = static_cast<MY_SIZE>(
      std::ceil(double(max_thread_num) / static_cast<double>(BLOCK_SIZE)));
  DataType *d_edge_weights;
  data_t<DataType> point_weights2(point_weights.getSize(),
                                  point_weights.getDim());
  std::copy(point_weights.begin(), point_weights.end(), point_weights2.begin());
  std::vector<MY_SIZE *> d_partition;
  for (const std::vector<MY_SIZE> &colour : partition) {
    MY_SIZE *d_colour;
    MY_SIZE mem_size = sizeof(MY_SIZE) * colour.size();
    checkCudaErrors(cudaMalloc((void **)&d_colour, mem_size));
    checkCudaErrors(
        cudaMemcpy(d_colour, colour.data(), mem_size, cudaMemcpyHostToDevice));
    d_partition.push_back(d_colour);
  }
  point_weights.initDeviceMemory();
  point_weights2.initDeviceMemory();
  checkCudaErrors(cudaMalloc((void **)&d_edge_weights,
                             sizeof(DataType) * graph.numEdges()));
  checkCudaErrors(cudaMemcpy(d_edge_weights, edge_weights,
                             sizeof(DataType) * graph.numEdges(),
                             cudaMemcpyHostToDevice));
  graph.edge_to_node.initDeviceMemory();
  // Timer t;
  TIMER_START(t);
  for (MY_SIZE i = 0; i < num; ++i) {
    for (MY_SIZE c = 0; c < num_of_colours; ++c) {
      problem_stepGPU<Dim, SOA><<<num_blocks, BLOCK_SIZE>>>(
          point_weights.getDeviceData(), d_edge_weights,
          graph.edge_to_node.getDeviceData(), d_partition[c],
          point_weights2.getDeviceData(), partition[c].size(),
          graph.numPoints());
      checkCudaErrors(cudaDeviceSynchronize());
    }
    TIMER_TOGGLE(t);
    if (reset_every && i % reset_every == reset_every - 1) {
      reset();
      // Copy to point_weights2 that is currently holding the result, the next
      // copy will put it into point_weights also.
      std::copy(point_weights.begin(), point_weights.end(),
                point_weights2.begin());
      point_weights2.flushToDevice();
    }
    checkCudaErrors(cudaMemcpy(
        point_weights.getDeviceData(), point_weights2.getDeviceData(),
        sizeof(DataType) * graph.numPoints() * Dim, cudaMemcpyDeviceToDevice));
    TIMER_TOGGLE(t);
  }
  PRINT_BANDWIDTH(
      t, "loopGPUEdgeCentred",
      sizeof(DataType) * (2.0 * Dim * graph.numPoints() + graph.numEdges()) *
          num,
      (sizeof(DataType) * graph.numPoints() * Dim * 2.0 + // point_weights
       sizeof(DataType) * graph.numEdges() * 1.0 +        // d_edge_weights
       sizeof(MY_SIZE) * graph.numEdges() * 2.0 +         // d_edge_list
       sizeof(MY_SIZE) * graph.numEdges() * 2.0           // d_partition
       ) * num);
  std::cout << " Needed " << num_of_colours << " colours" << std::endl;
  point_weights.flushToHost();
  checkCudaErrors(cudaFree(d_edge_weights));
  for (MY_SIZE i = 0; i < num_of_colours; ++i) {
    checkCudaErrors(cudaFree(d_partition[i]));
  }
}
/* 1}}} */

/* loopGPUHierarchical {{{1 */
template <unsigned Dim, bool SOA, typename DataType>
void Problem<Dim, SOA, DataType>::loopGPUHierarchical(MY_SIZE num,
                                                      MY_SIZE reset_every) {
  TIMER_START(t_colouring);
  HierarchicalColourMemory<Dim, SOA, DataType> memory(BLOCK_SIZE, *this);
  TIMER_PRINT(t_colouring, "Hierarchical colouring: colouring");
  const auto d_memory = memory.getDeviceMemoryOfOneColour();
  data_t<DataType> point_weights_out(point_weights.getSize(),
                                     point_weights.getDim());
  std::copy(point_weights.begin(), point_weights.end(),
            point_weights_out.begin());
  point_weights.initDeviceMemory();
  point_weights_out.initDeviceMemory();
  MY_SIZE total_cache_size = 0; // for bandwidth calculations
  DataType avg_num_edge_colours = 0;
  for (const typename HierarchicalColourMemory<
           Dim, SOA, DataType>::MemoryOfOneColour &memory_of_one_colour :
       memory.colours) {
    total_cache_size += memory_of_one_colour.points_to_be_cached.size();
    avg_num_edge_colours +=
        std::accumulate(memory_of_one_colour.num_edge_colours.begin(),
                        memory_of_one_colour.num_edge_colours.end(), 0.0f);
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
      MY_SIZE cache_size =
          sizeof(DataType) * d_memory[colour_ind].shared_size * Dim;
      problem_stepGPUHierarchical<Dim, SOA>
          <<<num_blocks, BLOCK_SIZE, cache_size>>>(
              static_cast<MY_SIZE *>(d_memory[colour_ind].edge_list),
              point_weights.getDeviceData(), point_weights_out.getDeviceData(),
              static_cast<DataType *>(d_memory[colour_ind].edge_weights),
              static_cast<MY_SIZE *>(d_memory[colour_ind].points_to_be_cached),
              static_cast<MY_SIZE *>(
                  d_memory[colour_ind].points_to_be_cached_offsets),
              static_cast<std::uint8_t *>(d_memory[colour_ind].edge_colours),
              static_cast<std::uint8_t *>(
                  d_memory[colour_ind].num_edge_colours),
              num_threads, graph.numPoints());
      checkCudaErrors(cudaDeviceSynchronize());
      total_num_blocks += num_blocks;
    }
    TIMER_TOGGLE(t);
    checkCudaErrors(cudaMemcpy(
        point_weights.getDeviceData(), point_weights_out.getDeviceData(),
        sizeof(DataType) * graph.numPoints() * Dim, cudaMemcpyDeviceToDevice));
    if (reset_every && iteration % reset_every == reset_every - 1) {
      reset();
      point_weights.flushToDevice();
      std::copy(point_weights.begin(), point_weights.end(),
                point_weights_out.begin());
      point_weights_out.flushToDevice();
    }
    TIMER_TOGGLE(t);
  }
  PRINT_BANDWIDTH(
      t, "GPU HierarchicalColouring",
      num * (2.0 * Dim * graph.numPoints() + graph.numEdges()) *
          sizeof(DataType),
      num * (sizeof(DataType) * graph.numPoints() * Dim * 2.0 + // point_weights
             sizeof(DataType) * graph.numEdges() * 1.0 +        // edge_weights
             sizeof(MY_SIZE) * graph.numEdges() * 2.0 +         // edge_list
             sizeof(MY_SIZE) * total_cache_size * 1.0 +
             sizeof(MY_SIZE) *
                 (total_num_blocks * 1.0 +
                  memory.colours.size()) + // points_to_be_cached_offsets
             sizeof(std::uint8_t) * graph.numEdges() // edge_colours
             ));
  std::cout << "  reuse factor: "
            << static_cast<double>(total_cache_size) / (2 * graph.numEdges())
            << std::endl;
  avg_num_edge_colours /=
      std::ceil(static_cast<double>(graph.numEdges()) / BLOCK_SIZE);
  std::cout << "  average number of colours used: " << avg_num_edge_colours
            << std::endl;
  // ---------------
  // -  Finish up  -
  // ---------------
  point_weights.flushToHost();
}
/* 1}}} */

template <unsigned Dim = 1, bool SOA = false, typename DataType = float>
using implementation_algorithm_t =
    void (Problem<Dim, SOA, DataType>::*)(MY_SIZE, MY_SIZE);

/* tests {{{1 */
template <unsigned Dim = 1, bool SOA = false, typename DataType = float>
void testTwoImplementations(
    MY_SIZE num, MY_SIZE N, MY_SIZE M, MY_SIZE reset_every,
    implementation_algorithm_t<Dim, SOA, DataType> algorithm1,
    implementation_algorithm_t<Dim, SOA, DataType> algorithm2) {
  std::cout << "========================================" << std:: endl;
  std::cout << "Two implementation test" << std::endl;
  std::cout << "Dim: " << Dim << ( SOA ? ", SOA": ", AOS" ) 
            << ", Precision: ";
  std::cout << ( sizeof(DataType) == sizeof(float) ? "float" : "double" );
  std::cout << std::endl << "Iteration: " << num << " size: " << N << ", " << M;
  std::cout << " reset: " << reset_every << std::endl;
  std::cout << "========================================" << std::endl;

  std::vector<DataType> result1, result2;
  std::vector<MY_SIZE> not_changed, not_changed2;
  DataType maxdiff = 0;
  MY_SIZE ind_max = 0, dim_max = 0;
  bool single_change_in_node = false;
  {
    srand(1);
    Problem<Dim, SOA, DataType> problem(N, M);
    result1.resize(problem.graph.numPoints() * Dim);
    //save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < Dim; ++d) {
        result1[i * Dim + d] = problem.point_weights[i * Dim + d];
      }
    }

    //run algorithm
    (problem.*algorithm1)(num, reset_every);
    
    DataType abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      MY_SIZE value_changed = Dim;
      for (MY_SIZE d = 0; d < Dim; ++d) {
        if(result1[i * Dim + d] == problem.point_weights[i * Dim + d]){
          if(value_changed == Dim)
            not_changed.push_back(i);
          value_changed--;
        }
        result1[i * Dim + d] = problem.point_weights[i * Dim + d];
        if (abs_max < problem.point_weights[i * Dim + d]) {
          abs_max = problem.point_weights[i * Dim + d];
          ind_max = i;
          dim_max = d;
        }
      }
      if(value_changed != Dim && value_changed != 0){
        single_change_in_node = true;
      }
    }
    std::cout << "Nodes stayed: " << not_changed.size() << "/" << problem.graph.numPoints() << std::endl;
    if(single_change_in_node){
      std::cout << "WARNING node values updated only some dimension." << std::endl; 
    }
    for(MY_SIZE i = 0; i < 10 && i < not_changed.size(); ++i){
      std::cout << "  " << not_changed[i]  << std::endl;
    }
    std::cout << "Abs max: " << abs_max << " node: " << ind_max << " dim: "
              << dim_max << std::endl;
  }

  MY_SIZE ind_diff = 0, dim_diff = 0;
  DataType max = 0;
  single_change_in_node = false;
  {
    srand(1);
    Problem<Dim, SOA, DataType> problem(N, M);
    result2.resize(problem.graph.numPoints() * Dim);
    //save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < Dim; ++d) {
        result2[i * Dim + d] = problem.point_weights[i * Dim + d];
      }
    }
    //run algorithm
    (problem.*algorithm2)(num, reset_every);
    DataType abs_max = 0;

    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      MY_SIZE value_changed = Dim;
      for (MY_SIZE d = 0; d < Dim; ++d) {
        if(result2[i * Dim + d] == problem.point_weights[i * Dim + d]){
          if(value_changed == Dim)
            not_changed2.push_back(i);
          value_changed--;
        }
        MY_SIZE ind = i * Dim + d;
        DataType diff = std::abs(problem.point_weights[ind] - result1[ind]) /
                        std::min(result1[ind], problem.point_weights[ind]);
        if (diff >= maxdiff) {
          maxdiff = diff;
          ind_diff = i;
          dim_diff = d;
          max = problem.point_weights[ind];
        }
        if( abs_max < problem.point_weights[ind] ){
          abs_max = problem.point_weights[ind]; 
          ind_max = i;
          dim_max = d;
        }
      }
      if(value_changed != Dim && value_changed != 0){
        std::cout << value_changed << " " << i << std::endl;
        for(MY_SIZE d = 0; d < Dim; ++d){
          std::cout << result2[i*Dim + d] << " / " << problem.point_weights[i*Dim+d] << "\t"; 
        }
        std::cout << std::endl;
        single_change_in_node = true;
      }
    }
    std::cout << "Nodes stayed: " << not_changed2.size() << "/" << problem.graph.numPoints() << std::endl;
    if(single_change_in_node){
      std::cout << "WARNING node values updated only some dimension." << std::endl; 
    }
    for(MY_SIZE i = 0; i < 10 && i < not_changed2.size(); ++i){
      std::cout << "  " << not_changed2[i]  << std::endl;
    }
    std::cout << "Abs max: " << abs_max << " node: " << ind_max << " dim: "
              << dim_max << std::endl;
    std::cout << "MAX DIFF: " << maxdiff << " node: " << ind_diff << " dim: "
              << dim_diff << std::endl;
    std::cout << "Values: " << result1[ind_diff * Dim + dim_diff] << " / " 
              << max << std::endl;
    std::cout << "Test considered " 
              << (maxdiff < 0.00001 ? "PASSED" : "FAILED")
              << std::endl;
  }
}

void testReordering(MY_SIZE num, MY_SIZE N, MY_SIZE M, MY_SIZE reset_every,
                    implementation_algorithm_t<> algorithm1,
                    implementation_algorithm_t<> algorithm2) {
  std::vector<float> result1;
  double rms = 0;
  {
    srand(1);
    Problem<> problem(N, M);
    /*std::ifstream f("test.in");*/
    /*Problem<> problem (f);*/
    std::cout << "Problem 1 created" << std::endl;
    problem.reorder();
    std::cout << "Problem 1 reordered" << std::endl;
    (problem.*algorithm1)(num, reset_every);
    float abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      result1.push_back(problem.point_weights[i]);
      abs_max = std::max(abs_max, std::abs(problem.point_weights[i]));
    }
    std::cout << "Abs max: " << abs_max << std::endl;
  }

  {
    srand(1);
    Problem<> problem(N, M);
    /*std::ifstream f("rotor37_mesh");*/
    /*Problem<> problem (f);*/
    std::cout << "Problem 2 created" << std::endl;
    (problem.*algorithm2)(num, reset_every);
    problem.reorder();
    float abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      rms += std::pow(problem.point_weights[i] - result1[i], 2);
      abs_max = std::max(abs_max, std::abs(problem.point_weights[i]));
    }
    std::cout << "Abs max: " << abs_max << std::endl;
    rms = std::pow(rms / result1.size(), 0.5);
    std::cout << "RMS: " << rms << std::endl;
  }
}
/* 1}}} */

template <unsigned Dim = 1, bool SOA = false, bool RunSerial = true,
          typename DataType = float>
void generateTimes(std::string in_file) {
  constexpr MY_SIZE num = 500;
  std::cout << ":::: Generating problems from file: " << in_file
            << "::::" << std::endl
            << "     Dimension: " << Dim << " SOA: " << std::boolalpha << SOA
            << std::endl;
  std::function<void(implementation_algorithm_t<Dim, SOA>, MY_SIZE)> run =
      [&in_file](implementation_algorithm_t<Dim, SOA> algo, MY_SIZE num) {
        std::ifstream f(in_file);
        Problem<Dim, SOA> problem(f);
        std::cout << "--Problem created" << std::endl;
        (problem.*algo)(num, 0);
        std::cout << "--Problem finished." << std::endl;
      };
  run(&Problem<Dim, SOA>::loopCPUEdgeCentred, RunSerial ? num : 1);
  run(&Problem<Dim, SOA>::loopCPUEdgeCentredOMP, num);
  run(&Problem<Dim, SOA>::loopGPUEdgeCentred, num);
  run(&Problem<Dim, SOA>::loopGPUHierarchical, num);
  std::cout << "Finished." << std::endl;
}

int main(int argc, const char **argv) {
  /*if (argc <= 1) {*/
  /*  std::cerr << "Usage: " << argv[0] << " <input graph>" << std::endl;*/
  /*  return 1;*/
  /*}*/
  /*generateTimes<1, true, false>(argv[1]);*/
  /*generateTimes<4, true, false>(argv[1]);*/
  /*generateTimes<8, true, false>(argv[1]);*/
  /*generateTimes<16, true, false>(argv[1]);*/
  MY_SIZE num = 500;
  MY_SIZE N = 1000, M = 2000;
  MY_SIZE reset_every = 0;
  #define TEST_DIM 4
  testTwoImplementations<TEST_DIM, false, float>(
      num, N, M, reset_every, &Problem<TEST_DIM, false, float>::loopCPUEdgeCentredOMP,
      &Problem<TEST_DIM, false, float>::loopGPUHierarchical);
  //testTwoImplementations<TEST_DIM, false, double>(
  //    num, N, M, reset_every, &Problem<TEST_DIM, false, double>::loopGPUEdgeCentred,
  //    &Problem<TEST_DIM, false, double>::loopCPUEdgeCentredOMP);

}

// vim:set et sw=2 ts=2 fdm=marker:
