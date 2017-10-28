#ifndef PROBLEM_HPP_CGW3IDMV
#define PROBLEM_HPP_CGW3IDMV

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

#include "grid.hpp"
#include "partition.hpp"
#include "timer.hpp"

constexpr MY_SIZE DEFAULT_BLOCK_SIZE = 128;

template <bool SOA, class ForwardIterator>
size_t countCacheLinesForBlock(ForwardIterator block_begin,
                               ForwardIterator block_end, MY_SIZE dim,
                               unsigned type_size);
template <bool SOA> struct HierarchicalColourMemory;

template <bool SOA = false> struct Problem {
  Mesh mesh;
  std::vector<data_t> point_weights;
  std::vector<data_t> cell_weights;
  const MY_SIZE block_size; // GPU block size
  std::vector<MY_SIZE> partition_vector;

  /* ctor/dtor {{{1 */
protected:
  Problem(Mesh &&mesh_,
          const std::vector<std::pair<MY_SIZE, unsigned>> &point_data_params,
          const std::vector<std::pair<MY_SIZE, unsigned>> &cell_data_params,
          MY_SIZE block_size_)
      : mesh(std::move(mesh_)), point_weights{}, cell_weights{},
        block_size{block_size_} {
    assert(point_data_params.size() == mesh.numMappings());
    for (unsigned mapping_ind = 0; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      point_weights.emplace_back(mesh.numPoints(mapping_ind),
                                 point_data_params[mapping_ind].first,
                                 point_data_params[mapping_ind].second);
    }
    for (unsigned i = 0; i < cell_data_params.size(); ++i) {
      cell_weights.emplace_back(mesh.numCells(), cell_data_params[i].first,
                                cell_data_params[i].second);
    }
  }

public:
  Problem(const std::vector<std::istream *> &mesh_is,
          const std::vector<MY_SIZE> &mesh_dim,
          const std::vector<std::pair<MY_SIZE, unsigned>> &point_data_params,
          const std::vector<std::pair<MY_SIZE, unsigned>> &cell_data_params,
          MY_SIZE _block_size = DEFAULT_BLOCK_SIZE)
      : mesh(mesh_is, mesh_dim), point_weights{}, cell_weights{},
        block_size{_block_size} {
    assert(point_data_params.size() == mesh.numMappings());
    for (unsigned mapping_ind = 0; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      point_weights.emplace_back(mesh.numPoints(mapping_ind),
                                 point_data_params[mapping_ind].first,
                                 point_data_params[mapping_ind].second);
    }
    for (unsigned i = 0; i < cell_data_params.size(); ++i) {
      cell_weights.emplace_back(mesh.numCells(), cell_data_params[i].first,
                                cell_data_params[i].second);
    }
  }

  ~Problem() {}

  Problem(Problem &&other)
      : mesh(std::move(other.mesh)),
        point_weights(std::move(other.point_weights)),
        cell_weights(std::move(other.cell_weights)),
        block_size(other.block_size),
        partition_vector(std::move(other.partition_vector)) {}
  /* 1}}} */

  /* loopGPUCellCentred {{{1 */
  template <class UserFunc> void loopGPUCellCentred(MY_SIZE num) {
    std::vector<std::vector<MY_SIZE>> partition = mesh.colourCells();
    MY_SIZE num_of_colours = partition.size();
    assert(num_of_colours > 0);
    data_t point_weights2(point_weights[0].getSize(), point_weights[0].getDim(),
                          point_weights[0].getTypeSize());
    std::copy(point_weights[0].begin(), point_weights[0].end(),
              point_weights2.begin());
    std::vector<std::vector<data_t>> d_cell_lists;
    std::vector<device_data_t> d_cell_to_node_ptrs;
    std::vector<std::vector<data_t>> d_cell_weights;
    std::vector<device_data_t> d_cell_data;
    MY_SIZE total_num_cache_lines = 0;
    MY_SIZE total_num_blocks = 0;
    for (const std::vector<MY_SIZE> &colour : partition) {
      d_cell_lists.emplace_back();
      d_cell_weights.emplace_back();
      std::vector<const MY_SIZE *> _cell_to_node;
      for (unsigned mapping_ind = 0; mapping_ind < mesh.numMappings();
           ++mapping_ind) {
        const unsigned mesh_dim = mesh.cell_to_node[mapping_ind].getDim();
        d_cell_lists.back().emplace_back(
            data_t::create<MY_SIZE>(colour.size(), mesh_dim));
        for (std::size_t i = 0; i < colour.size(); ++i) {
          std::copy_n(mesh.cell_to_node[mapping_ind].begin<MY_SIZE>() +
                          mesh_dim * colour[i],
                      mesh_dim,
                      d_cell_lists.back().back().begin<MY_SIZE>() +
                          mesh_dim * i);
        }
        d_cell_lists.back().back().initDeviceMemory();
        _cell_to_node.push_back(
            d_cell_lists.back().back().getDeviceData<MY_SIZE>());
      }
      d_cell_to_node_ptrs.emplace_back(device_data_t::create(_cell_to_node));
      std::vector<const void *> _cell_data;
      for (unsigned cw_ind = 0; cw_ind < cell_weights.size(); ++cw_ind) {
        const MY_SIZE cell_dim = cell_weights[cw_ind].getDim();
        d_cell_weights.back().emplace_back(colour.size(), cell_dim,
                                           cell_weights[cw_ind].getTypeSize());
        for (std::size_t i = 0; i < colour.size(); ++i) {
          for (unsigned d = 0; d < cell_dim; ++d) {
            std::copy_n(
                cell_weights[cw_ind].begin() +
                    cell_weights[cw_ind].getTypeSize() *
                        index<true>(mesh.numCells(), colour[i], cell_dim, d),
                cell_weights[cw_ind].getTypeSize(),
                d_cell_weights.back().back().begin() +
                    cell_weights[cw_ind].getTypeSize() *
                        index<true>(colour.size(), i, cell_dim, d));
          }
        }
        d_cell_weights.back().back().initDeviceMemory();
        _cell_data.push_back(d_cell_weights.back().back().getDeviceData());
      }
      d_cell_data.emplace_back(device_data_t::create(_cell_data));
      MY_SIZE num_blocks = std::ceil(static_cast<double>(colour.size()) /
                                     static_cast<double>(block_size));
      total_num_blocks += num_blocks;
      for (MY_SIZE i = 0; i < num_blocks; ++i) {
        total_num_cache_lines += countCacheLinesForBlock<SOA>(
            d_cell_lists.back()[0].begin<MY_SIZE>() +
                mesh.cell_to_node[0].getDim() * block_size * i,
            d_cell_lists.back()[0].begin<MY_SIZE>() +
                mesh.cell_to_node[0].getDim() *
                    std::min<MY_SIZE>(colour.size(), block_size * (i + 1)),
            point_weights[0].getDim(), point_weights[0].getTypeSize());
      }
    }
    std::vector<const void *> point_data;
    for (data_t &pw : point_weights) {
      pw.initDeviceMemory();
      point_data.push_back(pw.getDeviceData());
    }
    device_data_t d_point_data(device_data_t::create(point_data));
    device_data_t d_point_stride(device_data_t::create(mesh.numPoints()));
    std::vector<const MY_SIZE *> cell_to_node;
    point_weights2.initDeviceMemory();
    CUDA_TIMER_START(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      for (MY_SIZE c = 0; c < num_of_colours; ++c) {
        MY_SIZE num_blocks =
            std::ceil(static_cast<double>(partition[c].size()) /
                      static_cast<double>(block_size));
        UserFunc::template call<SOA>(
            d_point_data, point_weights2.getDeviceData(), d_cell_data[c],
            d_cell_to_node_ptrs[c], partition[c].size(), d_point_stride,
            partition[c].size(), num_blocks, block_size);
        checkCudaErrors(cudaDeviceSynchronize());
      }
      TIMER_TOGGLE(t);
      checkCudaErrors(cudaMemcpy(
          point_weights[0].getDeviceData(), point_weights2.getDeviceData(),
          point_weights[0].getTypeSize() * mesh.numPoints(0) *
              point_weights[0].getDim(),
          cudaMemcpyDeviceToDevice));
      TIMER_TOGGLE(t);
    }
    PRINT_BANDWIDTH(t, "loopGPUCellCentred", calcDataSize() * num);
    PRINT("Needed " << num_of_colours << " colours");
    PRINT("average cache_line / block: "
          << static_cast<double>(total_num_cache_lines) / total_num_blocks);
    PRINT_BANDWIDTH(t, " -cache line",
                    num * (total_num_cache_lines * 32.0 * 2));
    point_weights[0].flushToHost();
  }
  /* 1}}} */

  /* loopGPUHierarchical {{{1 */
  template <class UserFunc> void loopGPUHierarchical(MY_SIZE num) {
    TIMER_START(t_colouring);
    HierarchicalColourMemory<SOA> memory(*this, partition_vector);
    TIMER_PRINT(t_colouring, "Hierarchical colouring: colouring");
    const auto d_memory = memory.getDeviceMemoryOfOneColour();
    data_t point_weights_out(point_weights[0].getSize(),
                             point_weights[0].getDim(),
                             point_weights[0].getTypeSize());
    std::copy(point_weights[0].begin(), point_weights[0].end(),
              point_weights_out.begin());
    point_weights_out.initDeviceMemory();
    MY_SIZE total_cache_size = 0; // for bandwidth calculations
    double avg_num_cell_colours = 0;
    MY_SIZE total_num_blocks = 0;
    MY_SIZE total_shared_size = 0;
    size_t total_num_cache_lines = 0;
    for (MY_SIZE i = 0; i < memory.colours.size(); ++i) {
      const typename HierarchicalColourMemory<SOA>::MemoryOfOneColour
          &memory_of_one_colour = memory.colours[i];
      assert(memory.colours[i].cell_list[0].size() %
                 mesh.cell_to_node[0].getDim() ==
             0);
      MY_SIZE num_threads = memory_of_one_colour.cell_list[0].size() /
                            mesh.cell_to_node[0].getDim();
      MY_SIZE num_blocks = static_cast<MY_SIZE>(
          std::ceil(static_cast<double>(num_threads) / block_size));
      total_cache_size += memory_of_one_colour.points_to_be_cached.size();
      avg_num_cell_colours +=
          std::accumulate(memory_of_one_colour.num_cell_colours.begin(),
                          memory_of_one_colour.num_cell_colours.end(), 0.0f);
      total_num_blocks += num_blocks;
      total_shared_size += num_blocks * d_memory[i].shared_size;
      for (MY_SIZE j = 0;
           j < memory_of_one_colour.points_to_be_cached_offsets.size() - 1;
           ++j) {
        total_num_cache_lines +=
            countCacheLinesForBlock<SOA, std::vector<MY_SIZE>::const_iterator>(
                memory_of_one_colour.points_to_be_cached.begin() +
                    memory_of_one_colour.points_to_be_cached_offsets[j],
                memory_of_one_colour.points_to_be_cached.begin() +
                    memory_of_one_colour.points_to_be_cached_offsets[j + 1],
                point_weights[0].getDim(), point_weights[0].getTypeSize());
      }
    }
    std::vector<const char *> point_data(mesh.numMappings());
    std::transform(point_weights.begin(), point_weights.end(),
                   point_data.begin(), [](data_t &a) {
                     a.initDeviceMemory();
                     return a.getDeviceData();
                   });
    device_data_t d_point_data(device_data_t::create(point_data));
    device_data_t d_point_stride(device_data_t::create(mesh.numPoints()));
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
        MY_SIZE num_threads = memory.colours[colour_ind].cell_list[0].size() /
                              mesh.cell_to_node[0].getDim();
        MY_SIZE num_blocks = memory.colours[colour_ind].num_cell_colours.size();
        assert(num_blocks ==
               memory.colours[colour_ind].block_offsets.size() - 1);
        // + 32 in case it needs to avoid shared mem bank collisions
        MY_SIZE cache_size = point_weights[0].getTypeSize() *
                             (d_memory[colour_ind].shared_size + 32) *
                             point_weights[0].getDim();
        TIMER_TOGGLE(timer_calc);
        UserFunc::template call<SOA>(
            static_cast<const void **>(d_point_data),
            point_weights_out.getDeviceData(),
            static_cast<MY_SIZE *>(d_memory[colour_ind].points_to_be_cached),
            static_cast<MY_SIZE *>(
                d_memory[colour_ind].points_to_be_cached_offsets),
            static_cast<const void **>(d_memory[colour_ind].cell_weights.ptrs),
            static_cast<const MY_SIZE **>(d_memory[colour_ind].cell_list.ptrs),
            static_cast<std::uint8_t *>(d_memory[colour_ind].num_cell_colours),
            static_cast<std::uint8_t *>(d_memory[colour_ind].cell_colours),
            static_cast<MY_SIZE *>(d_memory[colour_ind].block_offsets),
            num_threads, d_point_stride, num_threads, num_blocks, block_size,
            cache_size);
        TIMER_TOGGLE(timer_calc);
        checkCudaErrors(cudaDeviceSynchronize());
      }
      assert(point_weights[0].getTypeSize() % sizeof(float) == 0);
      MY_SIZE copy_size = mesh.numPoints(0) * point_weights[0].getDim() *
                          point_weights[0].getTypeSize() / sizeof(float);
      TIMER_TOGGLE(timer_copy);
      MY_SIZE num_copy_blocks =
          std::ceil(static_cast<float>(copy_size) / 512.0);
      copyKernel<<<num_copy_blocks, 512>>>(
          reinterpret_cast<float *>(point_weights_out.getDeviceData()),
          reinterpret_cast<float *>(point_weights[0].getDeviceData()),
          copy_size);
      TIMER_TOGGLE(timer_copy);
    }
    PRINT_BANDWIDTH(timer_calc, "GPU HierarchicalColouring",
                    num * calcDataSize());
    PRINT_BANDWIDTH(timer_copy, " -copy",
                    2.0 * num * point_weights[0].getTypeSize() *
                        point_weights[0].getDim() * mesh.numPoints(0));
    PRINT("reuse factor: " << static_cast<double>(total_cache_size) /
                                  (mesh.cell_to_node[0].getDim() *
                                   mesh.numCells()));
    PRINT("cache/shared mem: " << static_cast<double>(total_cache_size) /
                                      total_shared_size);
    PRINT("shared mem reuse factor (total shared / (MeshDim * #cells)): "
          << static_cast<double>(total_shared_size) /
                 (mesh.cell_to_node[0].getDim() * mesh.numCells()));
    PRINT("average cache_line / block: "
          << static_cast<double>(total_num_cache_lines) / total_num_blocks);
    PRINT_BANDWIDTH(timer_calc, " -cache line",
                    num * (total_num_cache_lines * 32.0 * 2));
    avg_num_cell_colours /= total_num_blocks;
    PRINT("average number of colours used: " << avg_num_cell_colours);
    // ---------------
    // -  Finish up  -
    // ---------------
    point_weights[0].flushToHost();
  }
  /* 1}}} */

  template <class UserFunc> void stepCPUCellCentred(char *temp) const { /*{{{*/
    std::vector<const void *> point_data(point_weights.size());
    auto get_pointer = [](const data_t &a) { return a.cbegin(); };
    std::transform(point_weights.begin(), point_weights.end(),
                   point_data.begin(), get_pointer);
    std::vector<const void *> cell_data(cell_weights.size());
    std::transform(cell_weights.begin(), cell_weights.end(), cell_data.begin(),
                   get_pointer);
    std::vector<const MY_SIZE *> cell_to_node(mesh.numMappings());
    std::transform(mesh.cell_to_node.begin(), mesh.cell_to_node.end(),
                   cell_to_node.begin(),
                   [](const data_t &a) { return a.cbegin<MY_SIZE>(); });
    for (MY_SIZE cell_ind_base = 0; cell_ind_base < mesh.numCells();
         ++cell_ind_base) {
      UserFunc::template call<SOA>(point_data.data(), temp, cell_data.data(),
                                   cell_to_node.data(), cell_ind_base,
                                   mesh.numPoints().data(), mesh.numCells());
    }
  } /*}}}*/

  template <class UserFunc> void loopCPUCellCentred(MY_SIZE num) { /*{{{*/
    assert(point_weights.size() > 0);
    char *temp = (char *)malloc(point_weights[0].getTypeSize() *
                                mesh.numPoints(0) * point_weights[0].getDim());
    TIMER_START(t);
    TIMER_TOGGLE(t);
    std::copy(point_weights[0].begin(), point_weights[0].end(), temp);
    TIMER_TOGGLE(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      stepCPUCellCentred<UserFunc>(temp);
      TIMER_TOGGLE(t);
      std::copy_n(temp,
                  point_weights[0].getTypeSize() * mesh.numPoints(0) *
                      point_weights[0].getDim(),
                  point_weights[0].begin());
      TIMER_TOGGLE(t);
    }
    PRINT_BANDWIDTH(t, "loopCPUCellCentred", calcDataSize() * num);
    free(temp);
  } /*}}}*/

  template <class UserFunc>
  void stepCPUCellCentredOMP(const std::vector<MY_SIZE> &inds,
                             data_t &out) const { /*{{{*/
    std::vector<const void *> point_data(point_weights.size());
    auto get_pointer = [](const data_t &a) { return a.cbegin(); };
    std::transform(point_weights.begin(), point_weights.end(),
                   point_data.begin(), get_pointer);
    std::vector<const void *> cell_data(cell_weights.size());
    std::transform(cell_weights.begin(), cell_weights.end(), cell_data.begin(),
                   get_pointer);
    std::vector<const MY_SIZE *> cell_to_node(mesh.numMappings());
    std::transform(mesh.cell_to_node.begin(), mesh.cell_to_node.end(),
                   cell_to_node.begin(),
                   [](const data_t &a) { return a.cbegin<MY_SIZE>(); });
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < inds.size(); ++i) {
      MY_SIZE ind = inds[i];
      UserFunc::template call<SOA>(point_data.data(), out.begin(),
                                   cell_data.data(), cell_to_node.data(), ind,
                                   mesh.numPoints().data(), mesh.numCells());
    }
  } /*}}}*/

  template <class UserFunc> void loopCPUCellCentredOMP(MY_SIZE num) { /*{{{*/
    data_t temp(point_weights[0].getSize(), point_weights[0].getDim(),
                point_weights[0].getTypeSize());
    std::vector<std::vector<MY_SIZE>> partition = mesh.colourCells();
    MY_SIZE num_of_colours = partition.size();
    TIMER_START(t);
    TIMER_TOGGLE(t);
    // Init temp
    #pragma omp parallel for
    for (MY_SIZE e = 0;
         e < point_weights[0].getTypeSize() * point_weights[0].getSize() *
                 point_weights[0].getDim();
         ++e) {
      *(temp.begin() + e) = *(point_weights[0].begin() + e);
    }
    TIMER_TOGGLE(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      for (MY_SIZE c = 0; c < num_of_colours; ++c) {
        stepCPUCellCentredOMP<UserFunc>(partition[c], temp);
      }
      TIMER_TOGGLE(t);
      // Copy back from temp
      #pragma omp parallel for
      for (MY_SIZE e = 0;
           e < point_weights[0].getTypeSize() * point_weights[0].getSize() *
                   point_weights[0].getDim();
           ++e) {
        *(point_weights[0].begin() + e) = *(temp.begin() + e);
      }
      TIMER_TOGGLE(t);
    }
    PRINT_BANDWIDTH(t, "loopCPUCellCentredOMP", calcDataSize() * num);
  } /*}}}*/

  void reorder() {
    assert(mesh.numMappings() >= 1);
    ScotchReorder reorder(mesh.numPoints(0), mesh.numCells(),
                          mesh.cell_to_node[0]);
    std::vector<SCOTCH_Num> point_permutation = reorder.reorder();
    std::vector<MY_SIZE> inverse_permutation = mesh.reorder(point_permutation);
    reorderData<SOA>(point_weights[0], point_permutation);
    for (unsigned mapping_ind = 1; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      reorderData<SOA>(
          point_weights[mapping_ind],
          mesh.renumberPoints(mesh.getPointRenumberingPermutation(mapping_ind),
                              mapping_ind));
    }
    for (data_t &cw : cell_weights) {
      reorderDataInverse<true>(cw, inverse_permutation);
    }
  }

  void partition(float tolerance, idx_t options[METIS_NOPTIONS] = NULL) {
    std::vector<idx_t> _partition_vector;
    if (options == NULL) {
      idx_t _options[METIS_NOPTIONS];
      METIS_SetDefaultOptions(_options);
      _options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
      _partition_vector = std::move(partitionMetisEnh(
          mesh.getCellToCellGraph(), block_size, tolerance, options));
    } else {
      _partition_vector = std::move(partitionMetisEnh(
          mesh.getCellToCellGraph(), block_size, tolerance, options));
    }
    partition_vector.resize(_partition_vector.size());
    std::copy(_partition_vector.begin(), _partition_vector.end(),
              partition_vector.begin());
  }

  void reorderToPartition() {
    std::vector<MY_SIZE> inverse_permutation =
        mesh.reorderToPartition(partition_vector);
    for (data_t &cw : cell_weights) {
      reorderDataInverse<true>(cw, inverse_permutation);
    }
  }

  void renumberPoints() {
    for (unsigned mapping_ind = 0; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      std::vector<MY_SIZE> permutation = mesh.getPointRenumberingPermutation2(
          mesh.getPointToPartition(partition_vector, mapping_ind));
      mesh.renumberPoints(permutation, mapping_ind);
      reorderData<SOA>(point_weights[mapping_ind], permutation);
    }
  }

  void writePartition(std::ostream &os) const {
    os << block_size << std::endl;
    for (MY_SIZE p : partition_vector) {
      os << p << std::endl;
    }
  }

  void readPartition(std::istream &partition_is) {
    if (!partition_is) {
      throw InvalidInputFile{"partition input", 0, 0};
    }
    MY_SIZE read_block_size;
    partition_is >> read_block_size;
    if (!partition_is) {
      throw InvalidInputFile{"partition input", 0, 1};
    }
    if (read_block_size != block_size) {
      std::cerr << "Warning: block size in file (" << read_block_size
                << ") doesn't equal to used block size (" << block_size << ")"
                << std::endl;
    }
    partition_vector.resize(mesh.numCells());
    for (MY_SIZE i = 0; i < mesh.numCells(); ++i) {
      partition_is >> partition_vector[i];
      if (!partition_is) {
        throw InvalidInputFile{"partition input", 0, i + 1};
      }
    }
  }

  template <class DataType>
  void readPointData(std::istream &is, unsigned mapping_ind) {
    assert(mapping_ind < mesh.numMappings());
    assert(point_weights[mapping_ind].getTypeSize() == sizeof(DataType));
    if (!is) {
      throw InvalidInputFile{"point data input", mapping_ind, 0};
    }
    for (MY_SIZE i = 0; i < mesh.numPoints(mapping_ind); ++i) {
      for (MY_SIZE j = 0; j < point_weights[mapping_ind].getDim(); ++j) {
        is >> point_weights[mapping_ind].operator[]<DataType>(
                  index<SOA>(point_weights[mapping_ind].getSize(), i,
                             point_weights[mapping_ind].getDim(), j));
      }
      if (!is) {
        throw InvalidInputFile{"point data input", mapping_ind, i};
      }
    }
  }

  template <class DataType> void readCellData(std::istream &is, unsigned ind) {
    assert(ind < cell_weights.size());
    assert(cell_weights[ind].getTypeSize() == sizeof(DataType));
    if (!is) {
      throw InvalidInputFile{"cell data input", ind, 0};
    }
    for (MY_SIZE i = 0; i < mesh.numCells(); ++i) {
      for (MY_SIZE j = 0; j < cell_weights[ind].getDim(); ++j) {
        is >> cell_weights[ind].operator[]<DataType>(
                  j *cell_weights[ind].getSize() + i);
      }
      if (!is) {
        throw InvalidInputFile{"cell data input", ind, i};
      }
    }
  }

  double calcDataSize() const {
    double result = 0;
    for (unsigned mapping_ind = 0; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      result += point_weights[mapping_ind].getTypeSize() *
                point_weights[mapping_ind].getDim() *
                point_weights[mapping_ind].getSize();
      result += mesh.cell_to_node[mapping_ind].getTypeSize() *
                mesh.cell_to_node[mapping_ind].getDim() * mesh.numCells();
    }
    for (unsigned i = 0; i < cell_weights.size(); ++i) {
      result += cell_weights[i].getTypeSize() * cell_weights[i].getDim() *
                cell_weights[i].getSize();
    }
    result += point_weights[0].getTypeSize() * point_weights[0].getDim() *
              point_weights[0].getSize();
    return result;
  }
};

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

#endif /* end of include guard: PROBLEM_HPP_CGW3IDMV */
// vim:set et sw=2 ts=2 fdm=marker:
