#ifndef SIMULATION_HPP
#define SIMULATION_HPP
#define TIMER_MACRO

#include "timer.hpp"
#include <string>
#include <vector>

struct Kernel{
  std::string name;
  size_t count, byte_per_iter;
  TIMER(t)
  Kernel();
  Kernel( std::string _name, size_t _data_per_iter);

  void timerStart();
  void timerStop();

  float getBW();

  float getTime();
};

struct Simulation{
  std::vector<Kernel> kernels;
  std::vector<timer> timers;
  TIMER(total)
  
  Simulation();
  Simulation(std::vector<Kernel> _kernels, std::vector<timer> _timers);
  Simulation(size_t nedge, size_t nnode, int node_dim=1);

  void start();

  void stop();

  void printTiming();

};

Simulation initSimulation(size_t nedge, size_t nnode, int node_dim=1);

#endif
