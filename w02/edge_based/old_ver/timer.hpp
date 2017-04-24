#ifndef TIMER_HPP
#define TIMER_HPP

#ifdef TIMER_MACRO
  #define TIMER(t) timer t;
  #define TIMER_TOTAL(t) (t.total_time/1000000)
  #define TIMER_START(t) t.start();
  #define TIMER_STOP(t)  t.stop();
  #define TIMER_PRINT(t) t.print();
#else
  #define TIMER(t)
  #define TIMER_TOTAL(t) 0.0f
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

  timer(std::string _name):name(_name), total_time(0.0){}
  timer():name(""), total_time(0.0){}

  void start(){
    gettimeofday(&tv,NULL);
    total_time -= (double)tv.tv_sec*1000000+tv.tv_usec;
  }
  void stop(){
    gettimeofday(&tv,NULL);
    total_time += (double)tv.tv_sec*1000000+tv.tv_usec;
  }

  void print(){
    printf("%s\t%lf\n",name.c_str(),total_time/1000000);
  }

};

#endif
