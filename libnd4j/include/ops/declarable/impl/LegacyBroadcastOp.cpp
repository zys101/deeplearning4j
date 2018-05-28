//
// Created by raver119 on 17.10.2017.
//

#include <ops/declarable/LegacyBroadcastOp.h>

#ifdef __CUDABLAS__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#endif


namespace nd4j {
    namespace ops {
        template <typename T>
        Nd4jStatus LegacyBroadcastOp<T>::validateAndExecute(Context<T> &block) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            std::vector<int> dims(*block.getIArguments());

            if (dims.size() > 1)
                std::sort(dims.begin(), dims.end());

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            LegacyOpExecutor<T>::execBroadcastOp(*block.launchContext(), opNum,  z, y, z, dims);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        template <typename T>
        LegacyBroadcastOp<T>::LegacyBroadcastOp() : LegacyOp<T>::LegacyOp(2) {
            //
        }

        template <typename T>
        LegacyBroadcastOp<T>::LegacyBroadcastOp(int opNum) : LegacyOp<T>::LegacyOp(2, opNum) {
            //
        }

        template <typename T>
        LegacyOp<T>* LegacyBroadcastOp<T>::clone() {
            return new LegacyBroadcastOp(this->_opNum);
        }

        /**
        *   If external NDArray wasn't specified - the same shape is returned by all broadcast ops.
        */
        template <typename T>
        ShapeList* LegacyBroadcastOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            auto inShape = inputShape->at(0);

            // FIXME: remove memcpy
            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), Nd4jLong);
            memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

            return SHAPELIST(newShape);
        }


        template class ND4J_EXPORT LegacyBroadcastOp<float>;
        template class ND4J_EXPORT LegacyBroadcastOp<float16>;
        template class ND4J_EXPORT LegacyBroadcastOp<double>;
    }
}
