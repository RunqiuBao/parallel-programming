/**
  @file spmv_coo_cuda.cc
  @brief y = A * x for coo with cuda
 */

/** 
    @brief the device procedure to initialize all elements of v with 
    a constant c
    @param (v) a vector
    @param (c) the value to initialize all v's elements with
*/
__global__ void init_const_dev(vec_t v, real c) {
   int k = blockDim.x * blockIdx.x + threadIdx.x;
   real * y_dev = vy.elems_dev; 
   if (k < v.n)
     y_dev[k] = c;
}

/** 
    @brief the device procedure to do spmv in coo format
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @details assume A, vx, vy must have their elems_dev already set.
*/


__global__ void spmv_coo_dev(sparse_t A, vec_t vx, vec_t vy) {
    //idx_t M = A.M;
    idx_t nnz = A.nnz;
    coo_elem_t * elems_dev = A.coo.elems_dev;
    real * x_dev = vx.elems_dev;
    real * y_dev = vy.elems_dev;
    //for (idx_t i = 0; i < M; i++) {
    //  y_dev[i] = 0.0;
    //}

    int k;
    k = blockDim.x * blockIdx.x + threadIdx.x; // thread id

    if(k < nnz) {
      coo_elem_t * e = elems_dev + k;
      idx_t i = e->i;
      idx_t j = e->j;
      real  a = e->a;
      real ax = a * x_dev[j];
      //y[i] += ax;
      atomicAdd(&y_dev[i], ax);
      //printf("calculated\n");
    }

}

/** 
    @brief y = A * x for coo with cuda
    @param (A) a sparse matrix in coo format
    @param (vx) a vector
    @param (vy) a vector
    @returns 1 if succeed, 0 if failed
*/
static int spmv_coo_cuda(sparse_t A, vec_t vx, vec_t vy) {
  /*fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:spmv_coo_cuda:\n"
          "write a code that performs SPMV for COO format in parallel\n"
          "using CUDA.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);*/
  int nb, bs;
  bs = 256;
  nb = (A.nnz + bs - 1)/bs;
  printf("gonna enter the kernel\n");
  init_const_dev<<<nb, bs>>>(vy, 0.0);
  to_host((void *)vy.elems, (void *)vy.elems_dev, vy.n * sizeof(real));
  
  spmv_coo_dev<<<nb, bs>>>(A, vx, vy);
  printf("out from kernel\n");
  to_host((void *)vy.elems, (void *)vy.elems_dev, vy.n * sizeof(real));
  printf("spmv calclulation ended\n");
  /* this is a serial code for your reference */
  /*idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  for (idx_t k = 0; k < nnz; k++) {
    coo_elem_t * e = elems + k;
    idx_t i = e->i;
    idx_t j = e->j;
    real  a = e->a;
    real ax = a * x[j];
    y[i] += ax;
  }*/
  return 1;
}
