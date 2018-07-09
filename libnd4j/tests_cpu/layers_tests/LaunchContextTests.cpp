//
// Created by raver on 5/19/2018.
//

#include "testlayers.h"
#include <pointercast.h>
#include <Nd4j.h>
#include <NDArray.h>
#include <array/LaunchContext.h>

using namespace nd4j;
using namespace nd4j::ops;

class LaunchContextTests : public testing::Test {

};

TEST_F(LaunchContextTests,  Basic_Test_1) {
    //
    LaunchContext* context;
    context = new LaunchContext;
    nd4j_printf("Context %p was created successfully.\n", context);
    ASSERT_TRUE(context != nullptr);
    delete context; // this line crashes
}
