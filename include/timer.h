#ifndef TIMER_H
#define TIMER_H

#include <iostream>

class Timer {
   private:
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point _start = Clock::now();
    Clock::time_point _end = Clock::now();

   public:
    void tick() { _start = Clock::now(); }

    void tock() { _end = Clock::now(); }

    double duration() const { return std::chrono::duration<double>(_end - _start).count(); }
};

#endif
