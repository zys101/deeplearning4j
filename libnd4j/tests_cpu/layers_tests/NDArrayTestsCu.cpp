//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 29.05.2018
//

#include "testlayers.h"
#include <NDArray.h>

using namespace nd4j;

//////////////////////////////////////////////////////////////////////
class NDArrayTestCu : public testing::Test {
public:    
    
};


//////////////////////////////////////////////////////////////////////
// just draft, will be rewritten and amplified !!!
TEST_F(NDArrayTestCu, test1) { 
    
    Nd4jLong cShapeInfo[8] = {2, 2, 2, 2, 1, 0, 1, 99};
    Nd4jLong fShapeInfo[8] = {2, 2, 2, 1, 2, 0, 1, 102};
    float buffer[4] = {1,2,3,4};

    NDArray<float> arr1(cShapeInfo, true);
    NDArray<float> arr2('f', {2, 2}, {1,2,3,4});
    NDArray<float> arr3('c', {2, 2});
    NDArray<float> arr4(buffer, 'c', {2,2}, nullptr);
    NDArray<float> arr5(&arr4, true);
    NDArray<float> arr6 = arr5;

    ASSERT_TRUE(true);    
}



