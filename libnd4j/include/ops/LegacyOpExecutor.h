//
// Created by raver on 5/23/2018.
//

#ifndef LIBND4J_LEGACYOPEXECUTOR_H
#define LIBND4J_LEGACYOPEXECUTOR_H

#include <NDArray.h>
#include <array/LaunchContext.h>
#include <vector>

namespace nd4j {
    /**
     * This class executes Legacy ops by ID.
     * Used in legacy ops wrappers for Graph
     *
     * @tparam T
     */
    template <typename T>
    class LegacyOpExecutor {
    private:
    public:
        // TODO: we must get rid of std::vector here
        static void execScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, T scalar, std::vector<T> &extras);
    };
}



#endif //LIBND4J_LEGACYOPEXECUTOR_H
