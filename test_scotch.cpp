#include <iostream>

#define MY_SIZE unsigned int

#include "graph.hpp"
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>

//int startNewProcess(const char *cmd, char *const argv[]) {
//  pid_t pid = fork();
//  switch (pid) {
//  case -1: // Error
//    std::cerr << "Starting new process failed. (Fork failed.)" << std::endl;
//    return 1;
//  case 0: // Child
//    execv(cmd, argv);
//    std::cerr << "Starting new process failed. (Execv failed.)" << std::endl;
//    return 2;
//  default: // Parent
//    int status;
//    while (!WIFEXITED(status)) {
//      waitpid(pid, &status, 0);
//    }
//    if (WEXITSTATUS(status)) {
//      std::cerr << "Subprocess exited with exit code: " << WEXITSTATUS(status)
//                << std::endl;
//      return 3;
//    }
//  }
//  return 0;
//}

int testReorderImplementation() {
  Graph g(2, 4);
  {
    std::ofstream fout("/tmp/sulan/graph.grf");
    g.writeGraph(fout);
  }
  char *cmd = (char *)"/home/software/scotch_5.1.12/bin/gord";
  char *const argv[] = {cmd,
                        (char *)"/tmp/sulan/pici.grf",
                        (char *)"/tmp/sulan/graph.ord",
                        (char *)"/tmp/sulan/graph.log",
                        (char *)"-vst",
                        nullptr};
  if (startNewProcess(cmd, argv)) {
    return 1;
  }
  {
    std::ifstream fin("/tmp/sulan/graph.ord");
    if (g.readScotchReordering(fin)) {
      std::cerr
          << "Some error has happened. (See how verbose an error message I am?"
          << std::endl;
    }
  }
  return 0;
}

int main() {
  Graph g(1000, 2000);
  {
    std::cout << "Writing default ordering" << std::endl << std::flush;
    std::ofstream f("/tmp/sulan/my_edge_list");
    g.writeEdgeList(f);
  }
  if (g.reorder()) {
    return 1;
  }
  {
    std::cout << "Writing scotch reordering" << std::endl << std::flush;
    std::ofstream f("/tmp/sulan/scotch_reordered_edge_list");
    g.writeEdgeList(f);
  }
  return 0;
}
