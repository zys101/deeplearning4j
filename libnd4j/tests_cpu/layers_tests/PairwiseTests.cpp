//
// Created by agibsonccc on 1/17/17.
//
#include "testinclude.h"
#include <reduce3.h>

class EqualsTest : public testing::Test {
public:
    Nd4jLong firstShapeBuffer[8] = {2,1,2,1,1,0,1,102};
    float data[2] = {1.0,7.0};
    Nd4jLong secondShapeBuffer[8] = {2,2,1,6,1,0,6,99};
    float dataSecond[12] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    int opNum = 4;
    float extraArgs[1] = {1e-6f};
    int dimension[1] = {2147483647};
    int dimensionLength = 1;
};


TEST_F(EqualsTest,Eps) {
    float val = functions::reduce3::Reduce3<float>::execScalar(opNum,
                                                               data,
                                                               firstShapeBuffer,
                                                               extraArgs,
                                                               dataSecond,
                                                               secondShapeBuffer);
    ASSERT_TRUE(val < 0.5);
}
