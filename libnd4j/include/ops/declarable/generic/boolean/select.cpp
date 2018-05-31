//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_select)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(select, 3, 1, false, 0, 0) {
            auto cond = INPUT_VARIABLE(0);
            auto x = INPUT_VARIABLE(1);
            auto y = INPUT_VARIABLE(2);

            REQUIRE_TRUE(x->isSameShape(y), 0, "Select: X and Y shape should be equal");
            if (x->isScalar()) {
                REQUIRE_TRUE(cond->isScalar(), 0, "Select: Condition should gave either equal shape to X/Y first dimension or to be scalar");

                auto z = OUTPUT_VARIABLE(0);

                T v = cond->getScalar(0)  == (T) 0.0f ? y->getScalar(0) : x->getScalar(0);

                z->putScalar(0, v);
            } else {
                bool same = cond->isSameShape(x);
                REQUIRE_TRUE(cond->isScalar() || cond->lengthOf() == x->sizeAt(0) || same, 0, "Select: Condition should gave either equal shape to X/Y first dimension or to be scalar");
                if (same) {
                    auto z = OUTPUT_VARIABLE(0);

                    for (int e = 0; e < cond->lengthOf(); e++) {
                        T v = cond->getScalar(e);
                        T r = v == (T) 0.0f ? y->getScalar(e) : x->getScalar(e);
                        z->putScalar(e, r);
                    }
                } else {
                    REQUIRE_TRUE(cond->lengthOf() == x->sizeAt(0), 0, "Condition length should be equal to the dim0 of x/y to act as TAD-mask, but got %d instead", cond->lengthOf());

                    auto z = OUTPUT_VARIABLE(0);

                    auto dims = ShapeUtils<T>::convertAxisToTadTarget(x->rankOf(), {0});
                    auto tadsX = NDArrayFactory<T>::allTensorsAlongDimension(x, dims);
                    auto tadsY = NDArrayFactory<T>::allTensorsAlongDimension(y, dims);
                    auto tadsZ = NDArrayFactory<T>::allTensorsAlongDimension(z, dims);

                    for (int e = 0; e < tadsX->size(); e++) {
                        T v = cond->getScalar(e);
                        
                        if (v == (T) 0.0f)
                            tadsZ->at(e)->assign(tadsY->at(e));
                        else
                            tadsZ->at(e)->assign(tadsX->at(e));
                    }

                    delete tadsX;
                    delete tadsY;
                    delete tadsZ;
                }
            }

            return Status::OK();
        }

        DECLARE_SHAPE_FN(select) {
            auto inShape = inputShape->at(1);

            Nd4jLong *newshape;
            COPY_SHAPE(inShape, newshape);

            return SHAPELIST(newshape);
        }
    }
}

#endif