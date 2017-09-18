#include "graph.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " in_file out_file" << std::endl;
    return 1;
  }
  try {
    std::ifstream f(argv[1]);
    if (!f) {
      std::cerr << "Error opening input file: " << argv[1] << std::endl;
      return 3;
    }
    Graph graph(f);
    graph.reorderScotch();
    std::ofstream f2(argv[2]);
    graph.writeEdgeList(f2);
  } catch (ScotchError &e) {
    std::cerr << "Error during Scotch reordering: " << e.errorCode << std::endl;
    return 2;
  } catch (InvalidInputFile &e) {
    std::cerr << "Error in input file (" << e.input_type << ", line = "
      << e.line << ")" << std::endl;
    return 4;
  }
  return 0;
}
