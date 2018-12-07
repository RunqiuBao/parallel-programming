/** 
    @file spmv_csr_cuda.cc
    @brief y = A * x for csr with cuda
*/

/** 
    @brief y = A * x for csr with cuda
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @returns 1 if succeed, 0 if failed
*/
__global__
void spmv_csr_dev(sparse_t A, vec_t vx, vec_t vy){

  int kid = blockDim.x * blockIdx.x + threadIdx.x;//runqiu:thread id
  idx_t M = A.M;
  idx_t * row_start = A.csr.row_start_dev;
  csr_elem_t * elems = A.csr.elems_dev;
  real * x = vx.elems_dev;
  real * y = vy.elems_dev;
  if(kid<M){
    idx_t start = row_start[kid];
    idx_t end = row_start[kid + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = elems + k;
      idx_t j = e->j;
      real a = e->a;
      atomicAdd(&y[kid], a * x[j]);
    }

  }
  //printf("one thread finish!");
}

static int spmv_csr_cuda(sparse_t A, vec_t vx, vec_t vy) {

  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:spmv_csr_cuda:\n"
          "write a code that performs SPMV with CSR format in parallel\n"
          "using CUDA.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  //exit(1);
  int bs = 256;
  int nb = (A.M + bs - 1) / bs;
  check_launch_error((vy_to_zero<<<nb,bs>>>(A, vy)));
  check_launch_error((spmv_csr_dev<<<nb,bs>>>(A, vx, vy)));    
  to_host((void*)vy.elems, (void*)vy.elems_dev, sizeof(real)*A.M); 
  printf("receive vy succeed!");

  return 1;
}

