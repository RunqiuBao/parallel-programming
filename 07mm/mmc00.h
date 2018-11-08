/* 
 * mmc00.h
 */
#include "mmc.h"

template<idx_t M,idx_t N,idx_t K,idx_t lda,idx_t ldb,idx_t ldc,idx_t dM,idx_t dN>
idx_t gemm(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C) {
  for (idx_t i = 0; i < M; i++) {
    for (idx_t j = 0; j < N; j++) {
      asm volatile("# loop begins");
      for (idx_t k = 0; k < K; k++) {
	C(i,j) += A(i,k) * B(k,j);
      }
      asm volatile("# loop ends");
    }
  }
  return M * N * K;
}