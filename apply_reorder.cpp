#include <fstream>
#include <iostream>
#include "graph.hpp"

int main (int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " in_file out_file" << std::endl;
    return 1;
  }
  try {
    std::ifstream f (argv[1]);
    if (!f) {
      std::cerr << "Error opening input file: " << argv[1] << std::endl;
      return 3;
    }
    Graph graph (f);
    graph.reorder();
    std::ofstream f2 (argv[2]);
    graph.writeEdgeList(f2);
  } catch (ScotchError &e) {
    std::cerr << "Error during Scotch reordering: " << e.errorCode << std::endl;
    return 2;
  }
  return 0;
}
