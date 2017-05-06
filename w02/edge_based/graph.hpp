#ifndef GRAPH_HPP_35BFQORK
#define GRAPH_HPP_35BFQORK

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define MY_SIZE unsigned int
struct Graph {
private:
  MY_SIZE num_points, num_edges;

public:
  MY_SIZE *edge_list;
  MY_SIZE *offsets = nullptr, *point_list = nullptr;

 
  /**
   * Constructs graph from stream.
   *
   * Format:
   *   - first line: num_points and num_edges ("\d+\s+\d+")
   *   - next num_edges line: an edge, denoted by two numbers, the start- and
   *     endpoint respectively ("\d+\s+\d+")
   * If the reading is broken for some reason, the succesfully read edges are
   * kept and num_edges is set accordingly.
   */
  Graph(std::istream &is) /*: N(0), M(0)*/ {
    is >> num_points >> num_edges;
    edge_list = (MY_SIZE *)malloc(sizeof(MY_SIZE) * 2 * numEdges());
    for (MY_SIZE i = 0; i < num_edges; ++i) {
      if (!is) {
        num_edges = i;
        break;
      }
      is >> edge_list[2 * i] >> edge_list[2 * i + 1];
    }
  }

  ~Graph() {
    free(point_list);
    free(offsets);
    free(edge_list);
  }



  MY_SIZE numEdges() const { return num_edges; }

  MY_SIZE numPoints() const { return num_points; }

};

#endif /* end of include guard: GRAPH_HPP_35BFQORK */
