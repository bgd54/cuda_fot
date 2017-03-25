#ifndef GRAPH_HPP_35BFQORK
#define GRAPH_HPP_35BFQORK

#include <cstdlib>
#include <cstring>
#include <vector>

struct Graph {
private:
  const MY_SIZE N, M; // num of rows/columns (of points)
  MY_SIZE num_edges;

public:
  MY_SIZE *edge_list;
  MY_SIZE *offsets, *point_list;

  /* Initialisation {{{1 */
  Graph(MY_SIZE N_, MY_SIZE M_) : N(N_), M(M_) {
    // num_edges = (N - 1) * M + N * (M - 1); // vertical + horizontal
    num_edges = 2 * ((N - 1) * M + N * (M - 1)); // to and fro
    edge_list = (MY_SIZE *)malloc(sizeof(MY_SIZE) * 2 * numEdges());
    fillEdgeList2();

    // TODO
    // offsets = (MY_SIZE *)malloc(sizeof(MY_SIZE) * (N * M + 1));
    // point_list = (MY_SIZE *)malloc(sizeof(MY_SIZE) * 2 * numEdges());
    // fillPointList();
    offsets = point_list = nullptr;
  }

  ~Graph() {
    free(point_list);
    free(offsets);
    free(edge_list);
  }

  void fillEdgeList() {
    MY_SIZE array_ind = 0, upper_point_ind = 0, lower_point_ind = M;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        edge_list[array_ind++] = lower_point_ind;
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = ++upper_point_ind;
        ++lower_point_ind;
      }
      edge_list[array_ind++] = lower_point_ind++;
      edge_list[array_ind++] = upper_point_ind++;
    }
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      edge_list[array_ind++] = upper_point_ind;
      edge_list[array_ind++] = ++upper_point_ind;
    }
  }

  void fillEdgeList2() {
    MY_SIZE array_ind = 0, upper_point_ind = 0, lower_point_ind = M;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        // up-down
        edge_list[array_ind++] = lower_point_ind;
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = lower_point_ind;
        // right-left
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = upper_point_ind + 1;
        edge_list[array_ind++] = upper_point_ind + 1;
        edge_list[array_ind++] = upper_point_ind;
        ++lower_point_ind;
        ++upper_point_ind;
      }
      // Last up-down
      edge_list[array_ind++] = lower_point_ind;
      edge_list[array_ind++] = upper_point_ind;
      edge_list[array_ind++] = upper_point_ind++;
      edge_list[array_ind++] = lower_point_ind++;
    }
    // Last horizontal
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      edge_list[array_ind++] = upper_point_ind;
      edge_list[array_ind++] = upper_point_ind + 1;
      edge_list[array_ind++] = upper_point_ind + 1;
      edge_list[array_ind++] = upper_point_ind;
      ++upper_point_ind;
    }
  }

  void fillPointList() {
    MY_SIZE point_ind = 0, list_ind = 0, edge_ind = 0;
    MY_SIZE prev_degree = 0;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      offsets[point_ind] = prev_degree;
      ++prev_degree;
      point_list[list_ind++] = edge_ind++;
      point_list[list_ind++] = point_ind + M;
      ++point_ind;
      for (MY_SIZE c = 0; c < M - 1; ++c) {
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
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      offsets[point_ind] = prev_degree;
      ++prev_degree;
      point_list[list_ind++] = edge_ind++;
      point_list[list_ind++] = point_ind - 1;
      ++point_ind;
    }
    offsets[point_ind] = prev_degree; // should be end of point_list
  }
  /* 1}}} */

  std::vector<std::vector<MY_SIZE>>
  colourEdges(MY_SIZE from = 0,
              MY_SIZE to = static_cast<MY_SIZE>(-1)) const {
    if (to > numEdges()) {
      to = numEdges();
    }
    // First fit
    // TODO optimize so the sets have roughly equal sizes
    //      ^ do we really need that in hierarchical colouring?
    std::vector<std::vector<MY_SIZE>> edge_partitions;
    std::vector<std::uint8_t> point_colours(N * M, 0);
    for (MY_SIZE i = from; i < to; ++i) {
      std::uint8_t colour = point_colours[edge_list[2 * i + 1]]++;
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

  MY_SIZE numEdges() const { return num_edges; }

  MY_SIZE numPoints() const { return N * M; }
};

#endif /* end of include guard: GRAPH_HPP_35BFQORK */
// vim:set et sw=2 ts=2 fdm=marker: