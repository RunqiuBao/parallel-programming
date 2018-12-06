/** 
    @file vec_norm2_cuda.cc
    @brief the device procedure to do vec_norm2 on the device
*/

/** 
    @brief the device procedure to do vec_norm2 on the device
    @param (v) a vector
    @param (s) a pointer to a device memory to put the result into
    @details assume v.elems_dev already set and s a proper pointer
    to a device memory
*/

typedef struct{
  real * s;
#ifdef __NVCC__
  real * s_dev;
#endif
}real_t;

__global__ void vec_norm2_dev(vec_t v, real * s) {

  int k;
  k = blockDim.x * blockIdx.x + threadIdx.x; // thread id
  printf("in norm kernel\n");
  real * x_dev = v.elems_dev;
  idx_t n = v.n;

  if(k < n) {
    //s += x[i] * x[i];
    atomicAdd(s, x_dev[k] * x_dev[k]);
  }

}

/** 
    @brief square norm of a vector in parallel with cuda
    @param (v) a vector
    @returns the square norm of v (v[0]^2 + ... + v[n-1]^2)
*/
static real vec_norm2_cuda(vec_t v) {
  /*fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:vec_norm2_cuda:\n"
          "write a code that computes square norm of a vector v\n"
          "using CUDA.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);*/
  printf("to calculate the norm\n");
  int nb, bs;
  real * s;
  real * s_dev;
  *s = 0.0;
  s_dev =(real *)dev_malloc(sizeof(real));
  to_dev((void *)s_dev, (void *)s, sizeof(real));

  nb = 256;
  bs = 1024;
  printf("debug0\n");
  vec_norm2_dev<<<nb, bs>>>(v, s_dev);
  printf("debug1\n");
  to_host((void *)s, (void *)s_dev, sizeof(real));
  printf("copied s to host\n"); 
  return *s;
  /*real s = 0.0;
  real * x = v.elems;
  idx_t n = v.n;
  for (idx_t i = 0; i < n; i++) {
    s += x[i] * x[i];
  }
  return s;*/
}

