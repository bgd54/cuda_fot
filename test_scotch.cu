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
  Graph graph({1153,1153},{0,0},true);
  std::ifstream f_("/home/asulyok/graphs/wave");
  /*Problem<> problem(f_,288);*/
  std::ofstream f  ("/home/asulyok/graphs/grid_1153x1153_row_major2.gps_coord");
  std::ofstream f2 ("/home/asulyok/graphs/grid_1153x1153_row_major2.gps");
  /*std::ofstream f3 ("/home/asulyok/graphs/wave.gps.metis_part");*/
  /*problem.*/graph.reorderScotch();
  /*problem.partition(1.01);*/
  /*problem.reorderToPartition();*/
  /*problem.renumberPoints();*/
  /*problem.*/graph.writeCoordinates(f);
  /*problem.*/graph.writeEdgeList(f2);
  /*problem.writePartition(f3);*/
  return 0;
}
