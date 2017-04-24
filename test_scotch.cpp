#include <iostream>

#include "graph.hpp"
#include <fstream>

int main() {
  Graph g(1000,2000);
  {
    std::cout << "Writing default ordering" << std::endl << std::flush;
    std::ofstream f("/tmp/sulan/my_edge_list");
    g.writeEdgeList(f);
  }
  g.reorder();
  {
    std::cout << "Writing scotch reordering" << std::endl << std::flush;
    std::ofstream f("/tmp/sulan/scotch_reordered_edge_list");
    g.writeEdgeList(f);
  }
  return 0;
}
