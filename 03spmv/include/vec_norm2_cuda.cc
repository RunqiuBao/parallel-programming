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
__global__ void vec_norm2_dev(real * velemsdev, idx_t * vn, real * s) {

  int k;
  k = blockDim.x * blockIdx.x + threadIdx.x; // thread id

  real * x_dev = velemsdev;
  idx_t n = *vn;
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
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:vec_norm2_cuda:\n"
          "write a code that computes square norm of a vector v\n"
          "using CUDA.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  //exit(1);

  int nb,bs;
  real *s;
  real temp=0.0;
  s=&temp;
  real *s_dev;
  s_dev=(real*)dev_malloc(sizeof(real));
  to_dev((void*)s_dev, (void*)s, sizeof(real));

  idx_t *temp3=&v.n;
  idx_t *temp3_dev;
  bs=256;
  nb=(v.n+bs-1)/bs;
  
  vec_to_dev(v);
  temp3_dev=(idx_t*)dev_malloc(sizeof(idx_t));
  to_dev((void*)temp3_dev, (void*)temp3, sizeof(idx_t));
  //check_api_error((cudaMalloc((void **)&temp3_dev, sizeof(idx_t))));
  //check_api_error((cudaMemcpy(temp3_dev, temp3, sizeof(idx_t), cudaMemcpyHostToDevice)));

  check_launch_error((vec_norm2_dev<<<nb,bs>>>(v.elems_dev, temp3_dev, s_dev)));
  real *temp2;
  temp2=(real*)malloc(sizeof(real));
  to_host((void*)temp2, (void*)s_dev, sizeof(real));
  //check_api_error((cudaMemcpy(temp2, s.elems_dev, sizeof(real),cudaMemcpyHostToDevice)));
  printf("s to dev, succeed!:%f",*temp2);

  return *temp2; 
  /*real s = 0.0;
  real * x = v.elems;
  idx_t n = v.n;
  for (idx_t i = 0; i < n; i++) {
    s += x[i] * x[i];
  }
  return s;*/
}
