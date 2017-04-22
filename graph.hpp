#ifndef GRAPH_HPP_35BFQORK
#define GRAPH_HPP_35BFQORK

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ostream>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

inline int startNewProcess(const char *cmd, char *const argv[]) {
  pid_t pid = fork();
  switch (pid) {
  case -1: // Error
    std::cerr << "Starting new process failed. (Fork failed.)" << std::endl;
    return 1;
  case 0: // Child
  {
    execv(cmd, argv);
    int err = errno;
    std::cerr << "Starting new process failed. (Execv failed.)"
              << " Errno: " << err << std::endl;
    return 2;
  }
  default: // Parent
    int status = 0;
    while (!WIFEXITED(status)) {
      waitpid(pid, &status, 0);
    }
    if (WEXITSTATUS(status)) {
      std::cerr << "Subprocess exited with exit code: " << WEXITSTATUS(status)
                << std::endl;
      return 3;
    }
  }
  return 0;
}

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
  colourEdges(MY_SIZE from = 0, MY_SIZE to = static_cast<MY_SIZE>(-1)) const {
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

  /**
   * Scotch graph file, format according to the user manual 5.1:
   * http://gforge.inria.fr/docman/view.php/248/7104/scotch_user5.1.pdf
   */
  void writeGraph(std::ostream &os) const {
    os << 0 << std::endl; // Version number
    os << numPoints() << " " << numEdges() << std::endl;
    os << 0 << " "; // Base index
    os << "000";    // Flags: no vertex weights, no edge weights, no labels
    std::vector<std::pair<MY_SIZE, MY_SIZE>> g = getSortedEdges();
    MY_SIZE vertex = 0;
    std::vector<MY_SIZE> neighbours;
    for (const auto &edge : g) {
      if (edge.first != vertex) {
        assert(vertex < edge.first);
        os << std::endl << neighbours.size();
        for (MY_SIZE n : neighbours) {
          os << " " << n;
        }
        neighbours.clear();
        for (++vertex; vertex < edge.first; ++vertex) {
          os << std::endl << 0;
        }
      }
      neighbours.push_back(edge.second);
    }
    os << std::endl << neighbours.size();
    for (MY_SIZE n : neighbours) {
      os << " " << n;
    }
    neighbours.clear();
    for (++vertex; vertex < numPoints(); ++vertex) {
      os << std::endl << 0;
    }
  }

  /**
   * Writes the edgelist in the following format:
   *   - the first line contains two numbers separated by spaces, `numPoints()`
   *     and `numEdges()` respectively.
   *   - the following `numEdges()` lines contain two numbers, `i` and `j`,
   *     separated by spaces, and it means that there is an edge from `i` to `j`
   */
  void writeEdgeList(std::ostream &os) const {
    os << numPoints() << " " << numEdges() << std::endl;
    for (std::size_t i = 0; i < numEdges(); ++i) {
      os << edge_list[2 * i] << " " << edge_list[2 * i + 1] << std::endl;
    }
  }

  /**
   * Reads from a Scotch reordering file and reorders the edge list accordingly.
   */
  int readScotchReordering(std::istream &is) {
    {
      MY_SIZE file_num_points;
      is >> file_num_points;
      if (file_num_points != numPoints()) {
        return 1;
      }
    }
    std::vector<MY_SIZE> reordering(numPoints());
    for (MY_SIZE i = 0; i < numPoints(); ++i) {
      MY_SIZE from, to;
      is >> from >> to;
      reordering[from] = to;
    }
    std::for_each(edge_list, edge_list + 2 * numEdges(),
                  [&reordering](MY_SIZE &a) { a = reordering[a]; });
    std::vector<std::pair<MY_SIZE, MY_SIZE>> new_edge_list = getSortedEdges();
    for (MY_SIZE i = 0; i < numEdges(); ++i) {
      edge_list[2 * i + 0] = new_edge_list[i].first;
      edge_list[2 * i + 1] = new_edge_list[i].second;
    }
    return 0;
  }

  /**
   * Reorder using Scotch.
   */
  int reorder() {
    {
      std::ofstream fout("/tmp/sulan/graph.grf");
      writeGraph(fout);
    }
    char *cmd = (char *)"/home/software/scotch_5.1.12/bin/gord";
    char *const argv[] = {cmd,
                          (char *)"/tmp/sulan/graph.grf",
                          (char *)"/tmp/sulan/graph.ord",
                          (char *)"/tmp/sulan/graph.log",
                          (char *)"-vst",
                          nullptr};
    if (startNewProcess(cmd, argv)) {
      return 2;
    }
    {
      std::ifstream fin("/tmp/sulan/graph.ord");
      int status = readScotchReordering(fin);
      if (status) {
        std::cerr << "Some error has happened. (See how verbose an error "
                     "message I am?)"
                  << status << std::endl;
        return 1;
      }
    }
    return 0;
  }

private:
  std::vector<std::pair<MY_SIZE, MY_SIZE>> getSortedEdges() const {
    std::vector<std::pair<MY_SIZE, MY_SIZE>> g;
    for (MY_SIZE i = 0; i < numEdges(); ++i) {
      g.push_back(std::make_pair(edge_list[2 * i], edge_list[2 * i + 1]));
    }
    std::sort(g.begin(), g.end());
    return g;
  }
};

#endif /* end of include guard: GRAPH_HPP_35BFQORK */
// vim:set et sw=2 ts=2 fdm=marker:
