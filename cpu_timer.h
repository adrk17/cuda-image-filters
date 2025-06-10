#ifndef CPU_TIMER_H
#define CPU_TIMER_H

#include <chrono>

/**
 * Utility class for measuring CPU execution time using high-resolution clock.
 */
class CpuTimer {
public:
	/**
	 * Initializes the timer.
	 */ 
    void start() {
        m_start = std::chrono::high_resolution_clock::now();
    }

	/**
	 * Stops the timer.
	 */
    void stop() {
        m_stop = std::chrono::high_resolution_clock::now();
    }

	/**
	 * Returns elapsed time in milliseconds.
	 * @return Time in ms between start() and stop().
	 */
	double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(m_stop - m_start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_stop;
};

#endif