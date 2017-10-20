#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <set>
#include <vector>

#include "colouring.hpp"
#include "helper_cuda.h"
#include "kernels/mine.hpp"
#include "problem.hpp"
#include "tests.hpp"

/* copyKernels {{{1 */
__global__ void copyKernel(const float *__restrict__ a, float *__restrict__ b,
                           MY_SIZE size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const float4 *__restrict__ a_ = reinterpret_cast<const float4 *>(a);
  float4 *__restrict__ b_ = reinterpret_cast<float4 *>(b);
  if ((tid + 1) * 4 <= size) {
    b_[tid] = a_[tid];
  } else {
    for (MY_SIZE i = 0; i + tid * 4 < size; ++i) {
      b[4 * tid + i] = a[4 * tid + i];
    }
  }
}
/* 1}}} */

template <bool SOA = false, class ForwardIterator>
size_t countCacheLinesForBlock(ForwardIterator block_begin,
                               ForwardIterator block_end, MY_SIZE dim,
                               unsigned type_size) {
  std::set<MY_SIZE> cache_lines;
  MY_SIZE data_per_cacheline = 32 / type_size;

  for (; block_begin != block_end; ++block_begin) {
    MY_SIZE point_id = *block_begin;
    MY_SIZE cache_line_id = SOA ? point_id / data_per_cacheline
                                : point_id * dim / data_per_cacheline;
    if (!SOA) {
      if (data_per_cacheline / dim > 0) {
        assert(data_per_cacheline % dim == 0);
        cache_lines.insert(cache_line_id);
      } else {
        assert(dim % data_per_cacheline == 0);
        MY_SIZE cache_line_per_data =
            dim / data_per_cacheline; // Assume that Dim is multiple of
                                      // data_per_cacheline
        for (MY_SIZE i = 0; i < cache_line_per_data; ++i) {
          cache_lines.insert(cache_line_id++);
        }
      }
    } else {
      cache_lines.insert(cache_line_id);
    }
  }
  return (SOA ? dim : 1) * cache_lines.size();
}

/* loopGPUCellCentred {{{1 */
template <bool SOA>
template <class UserFunc>
void Problem<SOA>::loopGPUCellCentred(MY_SIZE num) {
  std::vector<std::vector<MY_SIZE>> partition = mesh.colourCells();
  MY_SIZE num_of_colours = partition.size();
  assert(num_of_colours > 0);
  data_t point_weights2(point_weights.getSize(), point_weights.getDim(),
                        point_weights.getTypeSize());
  std::copy(point_weights.begin(), point_weights.end(), point_weights2.begin());
  std::vector<data_t> d_cell_lists;
  std::vector<data_t> d_cell_weights;
  MY_SIZE total_num_cache_lines = 0;
  MY_SIZE total_num_blocks = 0;
  for (const std::vector<MY_SIZE> &colour : partition) {
    d_cell_lists.emplace_back(data_t::create<MY_SIZE>(colour.size(), MESH_DIM));
    d_cell_weights.emplace_back(colour.size(), cell_weights.getDim(),
                                cell_weights.getTypeSize());
    for (std::size_t i = 0; i < colour.size(); ++i) {
      std::copy_n(mesh.cell_to_node.begin<MY_SIZE>() + MESH_DIM * colour[i],
                  MESH_DIM,
                  d_cell_lists.back().begin<MY_SIZE>() + MESH_DIM * i);
      for (unsigned d = 0; d < cell_weights.getDim(); ++d) {
        std::copy_n(
            cell_weights.begin() +
                cell_weights.getTypeSize() *
                    index<true>(mesh.numCells(), colour[i],
                                cell_weights.getDim(), d),
            cell_weights.getTypeSize(),
            d_cell_weights.back().begin() +
                cell_weights.getTypeSize() *
                    index<true>(colour.size(), i, cell_weights.getDim(), d));
      }
    }
    d_cell_lists.back().initDeviceMemory();
    d_cell_weights.back().initDeviceMemory();
    MY_SIZE num_blocks = std::ceil(static_cast<double>(colour.size()) /
                                   static_cast<double>(block_size));
    total_num_blocks += num_blocks;
    for (MY_SIZE i = 0; i < num_blocks; ++i) {
      total_num_cache_lines += countCacheLinesForBlock<SOA>(
          d_cell_lists.back().begin<MY_SIZE>() + MESH_DIM * block_size * i,
          d_cell_lists.back().begin<MY_SIZE>() +
              MESH_DIM * std::min<MY_SIZE>(colour.size(), block_size * (i + 1)),
          point_weights.getDim(), point_weights.getTypeSize());
    }
  }
  point_weights.initDeviceMemory();
  point_weights2.initDeviceMemory();
  CUDA_TIMER_START(t);
  for (MY_SIZE i = 0; i < num; ++i) {
    for (MY_SIZE c = 0; c < num_of_colours; ++c) {
      MY_SIZE num_blocks = std::ceil(static_cast<double>(partition[c].size()) /
                                     static_cast<double>(block_size));
      UserFunc::template call<SOA>(
          point_weights.getDeviceData(), point_weights2.getDeviceData(),
          d_cell_weights[c].getDeviceData(),
          d_cell_lists[c].getDeviceData<MY_SIZE>(), partition[c].size(),
          mesh.numPoints(), partition[c].size(), num_blocks, block_size);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    TIMER_TOGGLE(t);
    checkCudaErrors(cudaMemcpy(
        point_weights.getDeviceData(), point_weights2.getDeviceData(),
        point_weights.getTypeSize() * mesh.numPoints() * point_weights.getDim(),
        cudaMemcpyDeviceToDevice));
    TIMER_TOGGLE(t);
  }
  PRINT_BANDWIDTH(
      t, "loopGPUCellCentred",
      (point_weights.getTypeSize() *
           (2.0 * point_weights.getDim() * mesh.numPoints() +
            cell_weights.getDim() * mesh.numCells()) +
       1.0 * MESH_DIM * sizeof(MY_SIZE) * mesh.numCells()) *
          num);
  std::cout << " Needed " << num_of_colours << " colours" << std::endl;
  std::cout << "  average cache_line / block: "
            << static_cast<double>(total_num_cache_lines) / total_num_blocks
            << std::endl;
  PRINT_BANDWIDTH(
      t, " -cache line",
      num * (total_num_cache_lines * 32.0 * 2 +
             1.0 * cell_weights.getDim() * mesh.numCells() *
                 point_weights.getDim() +
             1.0 * MESH_DIM * mesh.numCells() * sizeof(MY_SIZE)));
  point_weights.flushToHost();
}
/* 1}}} */

/* loopGPUHierarchical {{{1 */
template <bool SOA>
template <class UserFunc>
void Problem<SOA>::loopGPUHierarchical(MY_SIZE num) {
  TIMER_START(t_colouring);
  HierarchicalColourMemory<SOA> memory(*this, partition_vector);
  TIMER_PRINT(t_colouring, "Hierarchical colouring: colouring");
  const auto d_memory = memory.getDeviceMemoryOfOneColour();
  data_t point_weights_out(point_weights.getSize(), point_weights.getDim(),
                           point_weights.getTypeSize());
  std::copy(point_weights.begin(), point_weights.end(),
            point_weights_out.begin());
  point_weights.initDeviceMemory();
  point_weights_out.initDeviceMemory();
  MY_SIZE total_cache_size = 0; // for bandwidth calculations
  double avg_num_cell_colours = 0;
  MY_SIZE total_num_blocks = 0;
  MY_SIZE total_shared_size = 0;
  size_t total_num_cache_lines = 0;
  for (MY_SIZE i = 0; i < memory.colours.size(); ++i) {
    const typename HierarchicalColourMemory<SOA>::MemoryOfOneColour
        &memory_of_one_colour = memory.colours[i];
    MY_SIZE num_threads = memory_of_one_colour.cell_list.size() / MESH_DIM;
    MY_SIZE num_blocks = static_cast<MY_SIZE>(
        std::ceil(static_cast<double>(num_threads) / block_size));
    total_cache_size += memory_of_one_colour.points_to_be_cached.size();
    avg_num_cell_colours +=
        std::accumulate(memory_of_one_colour.num_cell_colours.begin(),
                        memory_of_one_colour.num_cell_colours.end(), 0.0f);
    total_num_blocks += num_blocks;
    total_shared_size += num_blocks * d_memory[i].shared_size;
    for (MY_SIZE j = 0;
         j < memory_of_one_colour.points_to_be_cached_offsets.size() - 1; ++j) {
      total_num_cache_lines +=
          countCacheLinesForBlock<SOA, std::vector<MY_SIZE>::const_iterator>(
              memory_of_one_colour.points_to_be_cached.begin() +
                  memory_of_one_colour.points_to_be_cached_offsets[j],
              memory_of_one_colour.points_to_be_cached.begin() +
                  memory_of_one_colour.points_to_be_cached_offsets[j + 1],
              point_weights.getDim(), point_weights.getTypeSize());
    }
  }
  // -----------------------
  // -  Start computation  -
  // -----------------------
  CUDA_TIMER_START(timer_calc);
  TIMER_TOGGLE(timer_calc);
  CUDA_TIMER_START(timer_copy);
  TIMER_TOGGLE(timer_copy);
  for (MY_SIZE iteration = 0; iteration < num; ++iteration) {
    for (MY_SIZE colour_ind = 0; colour_ind < memory.colours.size();
         ++colour_ind) {
      assert(memory.colours[colour_ind].cell_list.size() % MESH_DIM == 0);
      MY_SIZE num_threads =
          memory.colours[colour_ind].cell_list.size() / MESH_DIM;
      MY_SIZE num_blocks = memory.colours[colour_ind].num_cell_colours.size();
      assert(num_blocks == memory.colours[colour_ind].block_offsets.size() - 1);
      // + 32 in case it needs to avoid shared mem bank collisions
      MY_SIZE cache_size = point_weights.getTypeSize() *
                           (d_memory[colour_ind].shared_size + 32) *
                           point_weights.getDim();
      TIMER_TOGGLE(timer_calc);
      UserFunc::template call<SOA>(
          point_weights.getDeviceData(), point_weights_out.getDeviceData(),
          static_cast<MY_SIZE *>(d_memory[colour_ind].points_to_be_cached),
          static_cast<MY_SIZE *>(
              d_memory[colour_ind].points_to_be_cached_offsets),
          d_memory[colour_ind].cell_weights,
          static_cast<MY_SIZE *>(d_memory[colour_ind].cell_list),
          static_cast<std::uint8_t *>(d_memory[colour_ind].num_cell_colours),
          static_cast<std::uint8_t *>(d_memory[colour_ind].cell_colours),
          static_cast<MY_SIZE *>(d_memory[colour_ind].block_offsets),
          num_threads, mesh.numPoints(), num_threads, num_blocks, block_size,
          cache_size);
      TIMER_TOGGLE(timer_calc);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    assert(point_weights.getTypeSize() % sizeof(float) == 0);
    MY_SIZE copy_size = mesh.numPoints() * point_weights.getDim() *
                        point_weights.getTypeSize() / sizeof(float);
    TIMER_TOGGLE(timer_copy);
    MY_SIZE num_copy_blocks = std::ceil(static_cast<float>(copy_size) / 512.0);
    copyKernel<<<num_copy_blocks, 512>>>(
        reinterpret_cast<float *>(point_weights_out.getDeviceData()),
        reinterpret_cast<float *>(point_weights.getDeviceData()), copy_size);
    TIMER_TOGGLE(timer_copy);
  }
  PRINT_BANDWIDTH(
      timer_calc, "GPU HierarchicalColouring",
      num * ((2.0 * point_weights.getDim() * mesh.numPoints() +
              cell_weights.getDim() * mesh.numCells()) *
                 point_weights.getTypeSize() +
             1.0 * MESH_DIM * mesh.numCells() * sizeof(MY_SIZE)));
  PRINT_BANDWIDTH(timer_copy, " -copy",
                  2.0 * num * point_weights.getTypeSize() *
                      point_weights.getDim() * mesh.numPoints());
  std::cout << "  reuse factor: "
            << static_cast<double>(total_cache_size) /
                   (MESH_DIM * mesh.numCells())
            << std::endl;
  std::cout
      << "  cache/shared mem: "
      << static_cast<double>(total_cache_size) / total_shared_size
      << "\n  shared mem reuse factor (total shared / (MeshDim * #cells)): "
      << static_cast<double>(total_shared_size) / (MESH_DIM * mesh.numCells())
      << std::endl;
  std::cout << "  average cache_line / block: "
            << static_cast<double>(total_num_cache_lines) / total_num_blocks
            << std::endl;
  PRINT_BANDWIDTH(
      timer_calc, " -cache line",
      num * (total_num_cache_lines * 32.0 * 2 +
             1.0 * cell_weights.getDim() * mesh.numCells() *
                 point_weights.getTypeSize() +
             1.0 * MESH_DIM * mesh.numCells() * sizeof(MY_SIZE)));
  avg_num_cell_colours /= total_num_blocks;
  std::cout << "  average number of colours used: " << avg_num_cell_colours
            << std::endl;
  // ---------------
  // -  Finish up  -
  // ---------------
  point_weights.flushToHost();
}
/* 1}}} */

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
void generateTimesWithBlockDims(MY_SIZE N, MY_SIZE M,
                                std::pair<MY_SIZE, MY_SIZE> block_dims) {
  constexpr MY_SIZE num = 500;
  MY_SIZE block_size = block_dims.first == 0
                           ? block_dims.second
                           : block_dims.first * block_dims.second * 2;
  std::cout << ":::: Generating problems with block size: " << block_dims.first
            << "x" << block_dims.second << " (= " << block_size << ")"
            << "::::" << std::endl
            << "     Point dimension: " << PointDim
            << " Cell dimension: " << CellDim << " SOA: " << std::boolalpha
            << SOA << "\n     Data type: "
            << (sizeof(DataType) == sizeof(float) ? "float" : "double")
            << std::endl;
  std::function<void(implementation_algorithm_t<SOA>)> run =
      [&](implementation_algorithm_t<SOA> algo) {
        Problem<SOA> problem(
            std::move(StructuredProblem<2, PointDim, CellDim, SOA, DataType>(
                N, M, block_dims)));
        std::cout << "--Problem created" << std::endl;
        (problem.*algo)(num);
        std::cout << "--Problem finished." << std::endl;
      };
  run(&Problem<SOA>::template loopGPUCellCentred<
          mine::StepGPUGlobal2 < PointDim, CellDim, DataType>>);
  run(&Problem<SOA>::template loopGPUHierarchical<
          mine::StepGPUHierarchical2 < PointDim, CellDim, DataType>>);
  std::cout << "Finished." << std::endl;
}

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
void generateTimesDifferentBlockDims(MY_SIZE N, MY_SIZE M) {
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 32});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 8});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 4});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 128});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 32});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 16});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {8, 8});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 288});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 72});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 36});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {12, 12});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {9, 8});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 512});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 128});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 64});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {8, 32});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {16, 16});
}

void testReordering() {
  MY_SIZE num = 500;
  MY_SIZE N = 100, M = 200;
  constexpr unsigned TEST_DIM = 4;
  constexpr unsigned TEST_CELL_DIM = 4;
  testReordering<TEST_DIM, TEST_CELL_DIM, false, float>(
      num, N, M,
      &Problem<false>::loopCPUCellCentredOMP<MINE_KERNEL(StepOMP) < TEST_DIM,
                                             TEST_CELL_DIM, float>>,
      &Problem<false>::loopCPUCellCentredOMP<MINE_KERNEL(StepOMP) < TEST_DIM,
                                             TEST_CELL_DIM, float>>);
  testReordering<TEST_DIM, TEST_CELL_DIM, true, float>(
      num, N, M,
      &Problem<true>::loopCPUCellCentredOMP<MINE_KERNEL(StepOMP) < TEST_DIM,
                                            TEST_CELL_DIM, float>>,
      &Problem<true>::loopCPUCellCentredOMP<MINE_KERNEL(StepOMP) < TEST_DIM,
                                            TEST_CELL_DIM, float>>);
}

void testPartitioning() {
  MY_SIZE num = 500;
  MY_SIZE N = 100, M = 200;
  constexpr unsigned TEST_DIM = 4;
  constexpr unsigned TEST_CELL_DIM = 4;
  testPartitioning<TEST_DIM, TEST_CELL_DIM, false, float>(num, N, M);
  testPartitioning<TEST_DIM, TEST_CELL_DIM, true, float>(num, N, M);
}

void generateTimesDifferentBlockDims() {
  // SOA
  generateTimesDifferentBlockDims<1, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, true, float>(1153, 1153);
  // AOS
  generateTimesDifferentBlockDims<1, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, false, float>(1153, 1153);
  // SOA
  generateTimesDifferentBlockDims<1, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, true, double>(1153, 1153);
  // AOS
  generateTimesDifferentBlockDims<1, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, false, double>(1153, 1153);
}

int main(int argc, const char **argv) {
  /*generateTimesFromFile(argc, argv);*/
  testImplementations();
  /*testReordering();*/
  /*testPartitioning();*/
  /*generateTimesDifferentBlockDims();*/
  /*measurePartitioning();*/
  return 0;
}

// vim:set et sw=2 ts=2 fdm=marker:
