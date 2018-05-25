//
// Created by raver119 on 07.10.2017.
// Modified by GS <sgazeos@gmail.com> on 3/9/2018
//

#include <gemm.h>
#include <op_boilerplate.h>

namespace nd4j {
    namespace blas {

        template <typename T>
        int FORCEINLINE GEMM<T>::linearIndexC(int rows, int cols, int r, int c) {
            return (r * cols + c);
        }

        template <typename T>
        int FORCEINLINE GEMM<T>::linearIndexF(int rows, int cols, int r, int c) {
            return (c * rows + r);
        }

        template <typename T>
        T* GEMM<T>::transpose(int orderSource, int orderTarget, int rows, int cols, T *source) {
            
            return nullptr;
        }

        template <typename T>
        void GEMM<T>::op(int Order, int TransA, int TransB,
                       int M, int N, int K,
                       T alpha,
                       T *A, int lda,
                       T *B, int ldb,
                       T beta,
                       T *C, int ldc) {

        }


        template<typename T>
        void GEMV<T>::op(int TRANS, int M, int N,
                       T alpha,
                       T* A,
                       int lda,
                       T* X,
                       int incx,
                       T beta,
                       T* Y,
                       int incy ) {

        }


        template class ND4J_EXPORT GEMM<float>;
        template class ND4J_EXPORT GEMM<float16>;
        template class ND4J_EXPORT GEMM<double>;

        template class ND4J_EXPORT GEMV<float>;
        template class ND4J_EXPORT GEMV<float16>;
        template class ND4J_EXPORT GEMV<double>;
    }
}
