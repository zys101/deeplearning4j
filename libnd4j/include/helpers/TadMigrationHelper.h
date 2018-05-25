//
// Created by raver on 5/25/2018.
//

#ifndef LIBND4J_TADMIGRATIONHELPER_H
#define LIBND4J_TADMIGRATIONHELPER_H

#include <pointercast.h>
#include <helpers/TAD.h>

namespace nd4j {
    class TadMigrationHelper {
    private:
        Nd4jLong *_hostTadShapeInfo = nullptr;
        Nd4jLong *_hostTadOffsets = nullptr;
        Nd4jLong *_deviceTadShapeInfo = nullptr;
        Nd4jLong *_deviceTadOffsets = nullptr;
    public:
        explicit TadMigrationHelper(shape::TAD &tad);
        ~TadMigrationHelper();

        Nd4jLong* tadShapeInfo();
        Nd4jLong* tadOffsets();
    };
}


#endif //LIBND4J_TADMIGRATIONHELPER_H
