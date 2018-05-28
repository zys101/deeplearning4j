//
// Created by raver119 on 16.10.2017.
//

#include <ops/declarable/LegacyIndexReduceOp.h>
#include <helpers/ShapeUtils.h>

#ifdef __CUDABLAS__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#endif


namespace nd4j {
    namespace ops {


        template <typename T>
        LegacyIndexReduceOp<T>::LegacyIndexReduceOp() : LegacyOp<T>::LegacyOp(1){
            //
        }

        template <typename T>
        LegacyIndexReduceOp<T>::LegacyIndexReduceOp(int opNum) : LegacyOp<T>::LegacyOp(1, opNum) {
            //
        }

        template <typename T>
        LegacyOp<T>* LegacyIndexReduceOp<T>::clone() {
            return new LegacyIndexReduceOp(this->_opNum);
        }

        template <typename T>
        ShapeList *LegacyIndexReduceOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            auto inShape = inputShape->at(0);

            Nd4jLong *newShape;
            if (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT)) {
                // in this case we just return scalar
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
                newShape[0] = 2;
                newShape[1] = 1;
                newShape[2] = 1;
                newShape[3] = 1;
                newShape[4] = 1;
                newShape[5] = 0;
                newShape[6] = 1;
                newShape[7] = 99;
            } else {
                // in this case we're building proper shape for reduction
                auto array = new NDArray<T>(nullptr, inShape, block.getWorkspace());
                array->triggerAllocationFlag(false, false);

                newShape = ShapeUtils<T>::evalReduceShapeInfo('c', *block.getIArguments(), *array, false, true, block.workspace());

                delete array;
            }

            return SHAPELIST(newShape);
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        template <typename T>
        Nd4jStatus LegacyIndexReduceOp<T>::validateAndExecute(Context<T> &block) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            if (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT)) {
                // scalar
                LegacyOpExecutor<T>::execIndexReduceScalarOp(*block.launchContext(), opNum, x, z, *block.getTArguments());
            } else {
                // TAD
                std::vector<int> dims(*block.getIArguments());
                for (int e = 0; e < dims.size(); e++)
                    if (dims[e] < 0)
                        dims[e] += x->rankOf();

                if (dims.size() > 1)
                    std::sort(dims.begin(), dims.end());

                LegacyOpExecutor<T>::execIndexReduceOp(*block.launchContext(), opNum, x, z, dims, *block.getTArguments());
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LegacyIndexReduceOp<float>;
        template class ND4J_EXPORT LegacyIndexReduceOp<double>;
        template class ND4J_EXPORT LegacyIndexReduceOp<float16>;
    }
}