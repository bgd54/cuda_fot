#ifndef TIMER_HPP_UAYKBI6U
#define TIMER_HPP_UAYKBI6U

#define USE_TIMER_MACRO

#ifdef USE_TIMER_MACRO
#define TIMER_START(t) Timer t
#define TIMER_PRINT(t, pre)                                                    \
  do {                                                                         \
    std::cout << pre << " time: ";                                             \
    t.printTime();                                                             \
    std::cout << std::endl;                                                    \
  } while (0)
#define TIMER_TOGGLE(t) t.toggle()
#else
#define TIMER_START(t)
#define TIMER_PRINT(t)
#define TIMER_TOGGLE(t)
#endif

#include <chrono>
#include <iostream>

class Timer {
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point paused_time;
  bool paused = false;

public:
  Timer() : start{std::chrono::high_resolution_clock::now()} {}

  long long getTime() const {
    std::chrono::high_resolution_clock::time_point stop =
        std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
        .count();
  }

  void printTime() const { std::cout << getTime() << " ms"; }

  void toggle () {
    if (!paused) {
      paused_time = std::chrono::high_resolution_clock::now();
    } else {
      std::chrono::high_resolution_clock::time_point now =
          std::chrono::high_resolution_clock::now();
      start += now-paused_time;
    }
    paused = !paused;
  }

};

#endif /* end of include guard: TIMER_HPP_UAYKBI6U */
