#include <iostream>

#define MY_SIZE unsigned int

#include "graph.hpp"

int main () {
  Graph g (2,4);
  g.writeGraph(std::cout);
  std::cout << std::endl;
  return 0;
}
