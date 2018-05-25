//
// Created by raver on 5/25/2018.
//

#ifndef LIBND4J_VECTORMIGRATIONHELPER_H
#define LIBND4J_VECTORMIGRATIONHELPER_H

#include <pointercast.h>
#include <vector>

namespace nd4j {
    template <typename T>
    class VectorMigrationHelper {
    private:
        Nd4jLong _size = 0;
        T *_deviceData = nullptr;
    public:
        explicit VectorMigrationHelper(std::vector<T> &vector);
        ~VectorMigrationHelper();

        Nd4jLong size();
        T* data();
    };
}

#endif //LIBND4J_VECTORMIGRATIONHELPER_H
