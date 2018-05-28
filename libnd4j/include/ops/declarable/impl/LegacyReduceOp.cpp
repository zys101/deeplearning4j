//
// Created by raver119 on 16.10.2017.
//

#include <ops/declarable/LegacyReduceOp.h>
#include <helpers/TAD.h>
#include <helpers/ShapeUtils.h>

#ifdef __CUDABLAS__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#endif

namespace nd4j {
    namespace ops {
        template <typename T>
        LegacyReduceOp<T>::LegacyReduceOp() : LegacyOp<T>::LegacyOp(1) {
            //
        }

        template <typename T>
        LegacyReduceOp<T>::LegacyReduceOp(int opNum) : LegacyOp<T>::LegacyOp(1, opNum) {
            //this->_opNum = opNum;
        }

        template <typename T>
        LegacyOp<T>* LegacyReduceOp<T>::clone() {
            return new LegacyReduceOp(this->_opNum);
        }

        template <typename T>
        Nd4jStatus LegacyReduceOp<T>::validateAndExecute(Context<T> &block) {
            auto x = INPUT_VARIABLE(0);
        
            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();
            nd4j_debug("Executing LegacyReduceOp: [%i]\n", opNum);

            bool allAxes = false;

            if (block.width() == 1) {
                auto z = OUTPUT_VARIABLE(0);

                if (block.getIArguments()->size() == x->rankOf())
                    allAxes = true;

                if ((block.getIArguments()->size() == 0) ||
                    (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT) || allAxes) {
                    // scalar
                    LegacyOpExecutor<T>::execReduceScalarOp(*block.launchContext(), opNum, x, z, *block.getTArguments());
                } else {
                    // TAD
                    std::vector<int> dims(*block.getIArguments());

                    // normalizing axis
                    for (int e = 0; e < dims.size(); e++)
                        if (dims[e] < 0)
                            dims[e] += x->rankOf();

                    if (dims.size() > 1)
                        std::sort(dims.begin(), dims.end());

                    REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

                    LegacyOpExecutor<T>::execReduceOp(*block.launchContext(), opNum, x, z, dims, *block.getTArguments());
                }

                STORE_RESULT(*z);
            } else {
                auto indices = INPUT_VARIABLE(1);
                if (indices->lengthOf() == x->rankOf())
                    allAxes = true;

                std::vector<int> axis(indices->lengthOf());
                for (int e = 0; e < indices->lengthOf(); e++) {
                    // lol otherwise we segfault on macOS
                    auto f = static_cast<int>(indices->getScalar(e));
                    axis[e] = f >= 0 ? f : f += x->rankOf();
                }

                if ((block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT) || allAxes) {
                    auto z = OUTPUT_VARIABLE(0);

                    auto b = x->getBuffer();
                    auto s = x->shapeInfo();
                    auto e = block.numT() > 0 ? block.getTArguments()->data() : nullptr;

                    // scalar
                    T res = NativeOpExcutioner<T>::execReduceScalar(opNum, b, s, e);
                    z->putScalar(0, res);
                } else {
                    // TAD
                    if (indices->lengthOf() > 1)
                        std::sort(axis.begin(), axis.end());

                    REQUIRE_TRUE(!axis.empty(), 0, "Some dimensions required for reduction!");


                    auto newShape = ShapeUtils<T>::evalReduceShapeInfo(x->ordering(), axis, x);
                    auto z = new NDArray<T>(newShape, x->getWorkspace());

                    LegacyOpExecutor<T>::execReduceOp(*block.launchContext(), opNum, x, z, axis, *block.getTArguments());

                    RELEASE(newShape, x->getWorkspace());

                    // keepDims processing, for TF compatibility
                    if (block.getIArguments()->size() > 0 && block.getIArguments()->at(0) == 1) {
                        z->printShapeInfo("z shape before");
                        std::vector<Nd4jLong> newshape(z->getShapeAsVector());
                        for (int e = 0; e < axis.size(); e++) {
                            auto a = axis.at(e);
                            newshape.insert(newshape.begin() + a, 1);
                        }
                        z->reshapei(z->ordering(), newshape);
                        z->printShapeInfo("z shape after");
                    }

                    OVERWRITE_RESULT(z);
                }
            }

            return ND4J_STATUS_OK;
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        template <typename T>
        ShapeList *LegacyReduceOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            auto inShape = inputShape->at(0);

            Nd4jLong *newShape;

            bool allAxes = false;

            if (block.getIArguments()->size() == shape::rank(inShape))
                allAxes = true;

            if (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT) || allAxes) {
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

                newShape = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inShape), *block.getIArguments(), *array, false, false, block.workspace());

                delete array;
            }

            return SHAPELIST(newShape);
        }


        template class ND4J_EXPORT LegacyReduceOp<float>;
        template class ND4J_EXPORT LegacyReduceOp<float16>;
        template class ND4J_EXPORT LegacyReduceOp<double>;
    }
}