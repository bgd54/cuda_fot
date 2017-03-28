#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "timer.hpp"
#include <stdio.h>
#include <string>
#include <vector>
#include <cmath>

struct Kernel{
  std::string name;
  size_t count, byte_per_iter;
  TIMER(t)
  Kernel(){}
  Kernel( std::string _name, size_t _data_per_iter):name(_name), count(0),
          byte_per_iter(_data_per_iter) {}

  void timerStart(){
    ++count;
    TIMER_START(t)
  }
  void timerStop(){
    TIMER_STOP(t)
  }

  float getBW(){
    return byte_per_iter/TIMER_TOTAL(t)/pow(1024,3)*count; 
  }

  float getTime(){
    return TIMER_TOTAL(t);
  }
};

struct Simulation{
  std::vector<Kernel> kernels;
  std::vector<timer> timers;
  TIMER(total)
  
  Simulation(){}
  Simulation(std::vector<Kernel> _kernels, std::vector<timer> _timers):
          kernels(_kernels), timers(_timers){}

  void start(){
    TIMER_START(total)
  }

  void stop(){
    TIMER_STOP(total)
  }

  void printTiming(){
    for(timer&t:timers){
      TIMER_PRINT(t)
    }
    printf("\n  count   plan time     MPI time(std)        time(std)         "
            "  GB/s      GB/s   kernel name ");
    printf("\n "
            "-----------------------------------------------------------------"
            "--------------------------\n");
  for(size_t i=0; i<kernels.size(); ++i){
    float time = kernels[i].getTime();
    float bw = kernels[i].getBW();
    printf(" %6d;  %8.4f;  %8.4f(%8.4f);  %8.4f(%8.4f);  %8.4f;        "
                   " ;   %s \n",
                   (int)kernels[i].count, 0.0f, 0.0f, 0.0f, time, 0.0f, bw, 
                   kernels[i].name.c_str());
  }
  printf("Total plan time: %8.4f\n", 0.0f);
  printf("Max total runtime = %f\n", TIMER_TOTAL(total));

  }

};

Simulation initSimulation(size_t nedge, size_t nnode){
  std::vector<Kernel> kernels = 
        {Kernel("ssoln", 2*nnode*sizeof(float)),
          Kernel("iter", (2*nnode+nedge)*sizeof(float)), 
          Kernel("rms", 0)};
  std::vector<timer> timers;
  return Simulation(kernels, timers); 
}

#endif
