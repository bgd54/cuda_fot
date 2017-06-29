#include <iostream>

#include "graph.hpp"
#include "problem.hpp"
#include <fstream>

void testReordering() {
  Problem<> problem(2,4);
  std::cout << "Before reordering" << std::endl;
  std::cout << "Edges:" << std::endl;
  for (MY_SIZE i = 0; i < problem.graph.numEdges(); ++i) {
    std::cout << problem.graph.edge_to_node[2 * i] << "->"
              << problem.graph.edge_to_node[2 * i + 1] << "\t"
              << problem.edge_weights[i] << std::endl;
  }
  std::cout << "Points:" << std::endl;
  for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
    std::cout << problem.point_weights[i] << std::endl;
  }
  problem.reorder();
  std::cout << "After reordering" << std::endl;
  std::cout << "Edges:" << std::endl;
  for (MY_SIZE i = 0; i < problem.graph.numEdges(); ++i) {
    std::cout << problem.graph.edge_to_node[2 * i] << "->"
              << problem.graph.edge_to_node[2 * i + 1] << "\t"
              << problem.edge_weights[i] << std::endl;
  }
  std::cout << "Points:" << std::endl;
  for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
    std::cout << problem.point_weights[i] << std::endl;
  }
}

int main() {
  //testReordering();
  Graph graph(2049,1025,true);
  return 0;
}
