#ifndef CPU_TIMER_H
#define CPU_TIMER_H

#include <chrono>

class CpuTimer {
public:
    void start() {
        m_start = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        m_stop = std::chrono::high_resolution_clock::now();
    }

    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(m_stop - m_start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_stop;
};

#endif