#ifndef GRAPH_HPP_35BFQORK
#define GRAPH_HPP_35BFQORK

#include <cstdlib>
#include <vector>
#include <cstring>


struct Graph {
  const std::size_t N,M;  // num of rows/columns (of points)

  std::size_t *edge_list;
  std::size_t *offsets, *point_list;

  /* Initialisation {{{1 */
  Graph (std::size_t N_, std::size_t M_) : N(N_), M(M_) {
    edge_list = (std::size_t*)malloc(sizeof(std::size_t)*2*numEdges());
    fillEdgeList();

    offsets = (std::size_t*)malloc(sizeof(std::size_t)*(N*M+1));
    point_list = (std::size_t*)malloc(sizeof(std::size_t)*2*numEdges());
    fillPointList();
  }

  ~Graph () {
    free(point_list);
    free(offsets);
    free(edge_list);
  }


  void fillEdgeList () {
    std::size_t array_ind = 0, upper_point_ind = 0, lower_point_ind = M;
    for (std::size_t r = 0; r < N-1; ++r) {
      for (std::size_t c = 0; c < M-1; ++c) {
        edge_list[array_ind++] = lower_point_ind;
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = ++upper_point_ind;
        ++lower_point_ind;
      }
      edge_list[array_ind++] = lower_point_ind++;
      edge_list[array_ind++] = upper_point_ind++;
    }
    for (std::size_t c = 0; c < M-1; ++c) {
      edge_list[array_ind++] = upper_point_ind;
      edge_list[array_ind++] = ++upper_point_ind;
    }
  }

  void fillPointList () {
    std::size_t point_ind = 0, list_ind = 0, edge_ind = 0;
    std::size_t prev_degree = 0;
    for (std::size_t r = 0; r < N-1; ++r) {
      offsets[point_ind] = prev_degree; 
      ++prev_degree;
      point_list[list_ind++] = edge_ind++;
      point_list[list_ind++] = point_ind + M;
      ++point_ind;
      for (std::size_t c = 0; c < M-1; ++c) {
        offsets[point_ind] = prev_degree;
        prev_degree += 2;
        point_list[list_ind++] = edge_ind++;
        point_list[list_ind++] = point_ind - 1;
        point_list[list_ind++] = edge_ind++;
        point_list[list_ind++] = point_ind + M;
        ++point_ind;
      }
    }
    offsets[point_ind++] = prev_degree;
    for (std::size_t c = 0; c < M-1; ++c) {
      offsets[point_ind] = prev_degree;
      ++prev_degree;
      point_list[list_ind++] = edge_ind++;
      point_list[list_ind++] = point_ind - 1;
      ++point_ind;
    }
    offsets[point_ind] = prev_degree;  // should be end of point_list
  }
  /* 1}}} */

  std::vector<std::vector<std::size_t>> colourEdges (std::size_t from=0, 
      std::size_t to=static_cast<std::size_t>(-1)) const {
    if (to > numEdges()) {
      to = numEdges();
    }
    // First fit
    // TODO optimize so the sets have roughly equal sizes
    //      ^ do we really need that in hierarchical colouring?
    std::vector<std::vector<std::size_t>> edge_partitions;
    std::vector<std::uint8_t> point_colours (N*M,0);
    for (std::size_t i = from; i < to; ++i) {
      std::uint8_t colour = point_colours[edge_list[2*i+1]]++;
      if (colour == edge_partitions.size()) {
        edge_partitions.push_back({i});
      } else if (colour < edge_partitions.size()) {
        edge_partitions[colour].push_back(i);
      } else {
        // Wreak havoc
        std::cerr << "Something is wrong in the first fit algorithm in line "
          << __LINE__ << std::endl;
        std::terminate();
      }
    }
    return edge_partitions;
  }

  std::size_t numEdges() const {
    return (N-1)*M + N*(M-1);  // vertical + horizontal
  }
};

#endif /* end of include guard: GRAPH_HPP_35BFQORK */
// vim:set et sw=2 ts=2 fdm=marker:
