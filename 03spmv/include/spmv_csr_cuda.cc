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
void vy_to_zero(sparse_t A, vec_t vy){

  int kidcc = blockDim.x * blockIdx.x + threadIdx.x;//runqiu:thread id
  
  if(kidcc<A.M){
    //real temp=0.0;
    *(vy.elems_dev+kidcc)=0.0;
  }
  
}

__global__
void spmv_coo_dev(idx_t *ia, sparse_t A, vec_t vx, vec_t vy){

  int kidco = blockDim.x * blockIdx.x + threadIdx.x;//runqiu:thread id
  //idx_t * row_start = A.csr.row_start_dev;
  real * x = vx.elems_dev;
  real * y = vy.elems_dev; 
  //csr_elem_t * elems = A.csr.elems_dev;
  if(kidco<A.nnz){
    csr_elem_t * e = A.csr.elems_dev + kidco;
    idx_t j=e->j;
    real a=e->a;
    atomicAdd(&y[ia[kidco]], a * x[j]);
  }
  //printf("coo one step!");
}


__global__
void spmv_csr_dev(sparse_t A, vec_t vx, vec_t vy){

  int kid = blockDim.x * blockIdx.x + threadIdx.x;//runqiu:thread id
  idx_t M = A.M;
  idx_t * row_start = A.csr.row_start_dev;
  csr_elem_t * elems = A.csr.elems_dev;
  real * x = vx.elems_dev;
  real * y = vy.elems_dev;
  if(kid<M){
    //runqiu:tasks that every thread will do
    //y[kid]=0.0;
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
  
  /*runqiu:using cuda computing SPMV with CSR format*/
  if(1)//csr_to_dev(A) && vec_to_dev(vx) && vec_to_dev(vy)) //runqiu:transfer all the data to gpu, already in repeat_spmv
  {  
   // int bs = 256;
   // int nb = (A.nnz + bs - 1) / bs;
    if(A.M>50){
      int bs = 256;
      int nb = (A.M + bs - 1) / bs;
      check_launch_error((vy_to_zero<<<nb,bs>>>(A, vy)));
      check_launch_error((spmv_csr_dev<<<nb,bs>>>(A, vx, vy)));
      printf("spmv-csr succeed!");   
      //dev_free((void*)A.csr.elems_dev);
      //dev_free((void*)vx.elems_dev);
      //dev_free((void*)vy.elems_dev);
    }
    else 
    {
      int bs2 = 256;
      int bs=256;
      int nb=(A.M + bs - 1) / bs;
      int nb2 = (A.nnz + bs2 - 1) / bs2;
      idx_t *ia;
      ia=(idx_t*)malloc(A.nnz*sizeof(idx_t));
      //check_launch_error((csr_to_coo<<<nb2,bs2>>>(ia, A, vx, vy)));
      int k=0;
      for(int i=0;i<A.nnz;i++){
        if(k<(A.M-1))
          if(i<A.csr.row_start[k+1])
            ia[i]=i;
          else
            k++;
        else
          ia[i]=A.M;
      }
      printf("ia succeed!");
      idx_t *ia_dev;
      ia_dev=(idx_t*)dev_malloc(A.nnz*sizeof(idx_t));
      to_dev((void*)ia_dev, (void*)ia, sizeof(idx_t));
      printf("finish build coo!");
      check_launch_error((vy_to_zero<<<nb,bs>>>(A, vy)));
      check_launch_error((spmv_coo_dev<<<nb2,bs2>>>(ia_dev, A, vx, vy)));
      printf("spmv-csr succeed!");
      //dev_free((void*)A.csr.elems_dev);
      //dev_free((void*)ia);
      //dev_free((void*)vx.elems_dev);
    }
    to_host((void*)vy.elems, (void*)vy.elems_dev, sizeof(real)*A.M); 
    printf("receive vy succeed!");
  
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
