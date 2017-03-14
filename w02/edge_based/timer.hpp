#ifndef TIMER_HPP
#define TIMER_HPP

#ifdef TIMER_MACRO
  #define TIMER_START(t) t.start();
  #define TIMER_STOP(t)  t.stop();
  #define TIMER_PRINT(t) t.print();
#else
  #define TIMER_START(t)
  #define TIMER_STOP(t)
  #define TIMER_PRINT(t)
#endif

#include <sys/time.h>
#include <stdio.h>
#include <string>

struct timer{
  timeval tv;
  std::string name;
  double total_time;
  int count;

  timer(std::string _name):name(_name), total_time(0.0), count(0){}

  void start(){
    ++count;
    gettimeofday(&tv,NULL);
    total_time -= (double)tv.tv_sec*1000000+tv.tv_usec;
  }
  void stop(){
    gettimeofday(&tv,NULL);
    total_time += (double)tv.tv_sec*1000000+tv.tv_usec;
  }

  void print(){
    printf("%s\t%lf\t%d\n",name.c_str(),total_time/1000000,count);
  }


};

#endif
