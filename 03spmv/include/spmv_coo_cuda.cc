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

}

/** 
    @brief the device procedure to do spmv in coo format
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @details assume A, vx, vy must have their elems_dev already set.
*/
__global__
void vy_to_zero2(sparse_t A, vec_t vy){

  int kidcc = blockDim.x * blockIdx.x + threadIdx.x;//runqiu:thread id

  if(kidcc<A.M){
    //real temp=0.0;
    *(vy.elems_dev+kidcc)=0.0;
  }

}

__global__
void spmv_coo_dev2(sparse_t A, vec_t vx, vec_t vy){

  int kidco = blockDim.x * blockIdx.x + threadIdx.x;//runqiu:thread id
  //idx_t * row_start = A.csr.row_start_dev;
  real * x = vx.elems_dev;
  real * y = vy.elems_dev;
  //csr_elem_t * elems = A.csr.elems_dev;
  if(kidco<A.nnz){
    coo_elem_t * e = A.coo.elems_dev + kidco;
    idx_t j=e->j;
    idx_t i=e->i;
    real a=e->a;
    atomicAdd(&y[i], a * x[j]);
  }
  //printf("coo one step!");
}



/** 
    @brief y = A * x for coo with cuda
    @param (A) a sparse matrix in coo format
    @param (vx) a vector
    @param (vy) a vector
    @returns 1 if succeed, 0 if failed
*/
static int spmv_coo_cuda(sparse_t A, vec_t vx, vec_t vy) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:spmv_coo_cuda:\n"
          "write a code that performs SPMV for COO format in parallel\n"
          "using CUDA.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  //exit(1);

  /* this is a serial code for your reference 
  idx_t M = A.M;
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
   int bs2 = 256;
   int bs=256;
   int nb=(A.M + bs - 1) / bs;
   int nb2 = (A.nnz + bs2 - 1) / bs2;
   printf("start coospmv!");
   check_launch_error((vy_to_zero2<<<nb,bs>>>(A, vy)));
   check_launch_error((spmv_coo_dev2<<<nb2,bs2>>>(A, vx, vy)));
   to_host((void*)vy.elems, (void*)vy.elems_dev, sizeof(real)*A.M);
   return 1;
}
