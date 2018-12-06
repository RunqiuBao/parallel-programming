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

/*runqiu:kernel*/
__global__
void spmv_csr_dev(sparse_t & A, vec_t & vx, vec_t & vy){

  int kid = blockDim.x * blockIdx.x + threadIdx.x;//runqiu:thread id
  idx_t M = A.M;
  idx_t * row_start = A.csr.row_start_dev;
  csr_elem_t * elems = A.csr.elems_dev;
  real * x = vx.elems_dev;
  real * y = vy.elems_dev;
  if(kid<M){
    //runqiu:tasks that every thread will do
    y[kid] = 0.0;
    idx_t start = row_start[kid];
    idx_t end = row_start[kid + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = elems + k;
     //idx_t j = e->j;
      real  a = e->a;
      atomicAdd(&y[kid], a * x[k]);
    }

  }

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
  
  /*runqiu:using cuda computing SPMV with CSR format*/
  if(1)//csr_to_dev(A) && vec_to_dev(vx) && vec_to_dev(vy)) //runqiu:transfer all the data to gpu, already in repeat_spmv
  {  
    int bs = 256;
    int nb = (A.nnz + bs - 1) / bs;
    check_launch_error((spmv_csr_dev<<<nb,bs>>>(A, vx, vy)));
    printf("spmv-csr succeed!");
    //check_api_error(());
    //vy.elems = (real*)malloc(sizeof(real)*A.M);
    //check_api_error((cudaMemcpy(vy.elems, vy.elems_dev, sizeof(real)*A.M, cudaMemcpyDeviceToHost)));
  }

  /* this is a serial code for your reference
  idx_t M = A.M;
  idx_t * row_start = A.csr.row_start;
  csr_elem_t * elems = A.csr.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) { // conver to kernel + kernel launch
    y[i] = 0.0;
  }
  for (idx_t i = 0; i < M; i++) { // conver to kernel + kernel launch
    idx_t start = row_start[i];
    idx_t end = row_start[i + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = elems + k;
      idx_t j = e->j;
      real  a = e->a;
      y[i] += a * x[j];
    }
  } */
  return 1;
}

/*runqiu:nested kernel <https://blog.csdn.net/lingerlanlan/article/details/26258117>
__global__ spmv_csr_dev(A, vx, vy){
  
  kid = blockDim.x * blockIdx.x + threadIdx.x;//runqiu:thread id
  idx_t M = A.M;
  idx_t * row_start = A.csr.row_start_dev;
  csr_elem_t * elems = A.csr.elems_dev;
  real * x = vx.elems_dev;
  real * y = vy.elems_dev;
  if(kid<M){
    //runqiu:tasks that every thread will do
    y[kid] = 0.0;
    idx_t start = row_start[kid];
    idx_t end = row_start[kid + 1];
    int bs = 32;
    int nb = (end-start + bs - 1) / bs;
    nested_dev<<<bs,nb>>>(elems, y, end, start);
  }
  
}

__global__ nested_dev(elems, y){
  knid = blockDim.x * blockIdx.x + threadIdx.x;
  if(knid<lrow)
  {
    csr_elem_t * e = elems + start + knid;
    idx_t j = e->j;
    real  a = e->a;
    atomicAdd(&y[i], a * x[j]);
  }
}
*/
