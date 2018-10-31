/**
   @file cycle.h
   @brief a small procedure to get CPU/reference cycle
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <pthread.h>

/**
   this is a wrapper to Linux system call perf_event_open
 */
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                group_fd, flags);
  return ret;
}

/**
   @brief a structure encapsulating a performance counter
 */
typedef struct {
  pthread_t tid;                /**< thread ID this is valid for */
  int fd;                       /**< what perf_event_open returned  */
} cpu_clock_counter_t;

/**
   @brief make a cpu_clock_counter
   @details 
   cpu_clock_counter_t t = mk_cpu_clock_counter();
   long c0 = cpu_clock_counter_get(t);
      ... do something ...
   long c1 = cpu_clock_counter_get(t);
   long dc = c1 - c0; <- the number of CPU clocks between c0 and c1
  */
static cpu_clock_counter_t mk_cpu_clock_counter() {
  pthread_t tid = pthread_self();
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(struct perf_event_attr));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(struct perf_event_attr);
  //pe.config = PERF_COUNT_HW_INSTRUCTIONS;
  pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  
  int fd = perf_event_open(&pe, 0, -1, -1, 0);
  if (fd == -1) {
    perror("perf_event_open");
    exit(EXIT_FAILURE);
  }
  if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1) {
    perror("ioctl");
    exit(EXIT_FAILURE);
  }
  if (ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1) {
    perror("ioctl");
    exit(EXIT_FAILURE);
  }
  cpu_clock_counter_t cc = { tid, fd };
  return cc;
}

/**
   @brief destroy a cpu clock counter
  */
static void cpu_clock_counter_destroy(cpu_clock_counter_t cc) {
  close(cc.fd);
}

/**
   @brief get CPU clock
  */
static long long cpu_clock_counter_get(cpu_clock_counter_t cc) {
  pthread_t tid = pthread_self();
  if (tid != cc.tid) {
    fprintf(stderr,
            "%s:%d:cpu_clock_counter_get: the caller thread (%ld)"
            " is invalid (!= %ld)\n", __FILE__, __LINE__, tid, cc.tid);
    return -1;
  } else {
    long long c;
    ssize_t rd = read(cc.fd, &c, sizeof(long long));
    if (rd == -1) {
      perror("read"); 
      exit(EXIT_FAILURE);
    }
    assert(rd == sizeof(long long));
    return c;
  }
}

/**
   @brief read reference clock
  */
static inline long long rdtsc() {
  long long u;
  asm volatile ("rdtsc;shlq $32,%%rdx;orq %%rdx,%%rax":"=a"(u)::"%rdx");
  return u;
}

/**
   @brief get ns
  */
static inline long long cur_time_ns() {
  struct timespec ts[1];
  if (clock_gettime(CLOCK_REALTIME, ts) == -1) {
    perror("clock_gettime"); exit(1);
  }
  return ts->tv_sec * 1000000000L + ts->tv_nsec;
}

#if 0
int main(int argc, char **argv) {
  float a = (argc > 1 ? atof(argv[1]) : 0.9);
  float b = (argc > 2 ? atof(argv[2]) : 1.0);
  float n = (argc > 3 ? atol(argv[3]) : 1000000);
  float x = 1.0;
  cycle_timer_t tm = mk_cycle_timer_for_thread();
  cycle_t t0 = cycle_timer_get(tm);
  for (long i = 0; i < n; i++) {
    x = a * x + b;
  }
  cycle_t t1 = cycle_timer_get(tm);
  printf("%lld (%lld) CPU (REF) cyles %.4f (%.4f) CPU (REF) cycles/iter\n",
         (t1.c - t0.c),
         (t1.r - t0.r),
         (t1.c - t0.c)/(double)n,
         (t1.r - t0.r)/(double)n);
  printf("x = %f\n", x);
  
  close(tm.fd);
}

#endif
