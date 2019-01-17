/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 11.10.2017.
//

#include "testlayers.h"
#include <vector>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpTuple.h>
#include <ops/declarable/OpRegistrator.h>
#include <GraphExecutioner.h>
#include <memory/MemoryReport.h>
#include <memory/MemoryUtils.h>
#include <MmulHelper.h>

using namespace nd4j;
using namespace nd4j::ops;

class OneOffTests : public testing::Test {
public:

};

TEST_F(OneOffTests, test_avg_pool_3d_1) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/avg_pooling3d.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    delete graph;
}

TEST_F(OneOffTests, test_non2d_0A_1) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/non2d_0A.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    delete graph;
}

TEST_F(OneOffTests, test_assert_scalar_float32_1) {
    nd4j::ops::Assert op;
    nd4j::ops::identity op1;
    nd4j::ops::noop op2;
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/scalar_float32.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    delete graph;
}

TEST_F(OneOffTests, test_pad_1D_1) {
    auto e = NDArrayFactory::create<float>('c', {7}, {10.f,0.778786f, 0.801198f, 0.724375f, 0.230894f, 0.727141f,10.f});
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/pad_1D.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(4));

    auto z = graph->getVariableSpace()->getVariable(4)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    z->printIndexedBuffer("z");

    ASSERT_EQ(e, *z);
    delete graph;
}
/*
TEST_F(OneOffTests, test_scatter_nd_update_1) {

    auto e = NDArrayFactory::create<float>('c', {10, 7}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.20446908f, 0.37918627f, 0.99792874f, 0.71881700f, 0.18677747f,
                                                    0.78299069f, 0.55216062f, 0.40746713f, 0.92128086f, 0.57195139f, 0.44686234f, 0.30861020f, 0.31026053f, 0.09293187f,
                                                    1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.95073712f, 0.45613325f, 0.95149803f, 0.88341522f, 0.54366302f, 0.50060666f, 0.39031255f,
                                                    1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                                                    1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/scatter_nd_update.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(6));

    auto z = graph->getVariableSpace()->getVariable(6)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    z->printIndexedBuffer("z");

    ASSERT_EQ(e, *z);

    delete graph;
}
 */

TEST_F(OneOffTests, test_conv2d_nhwc_failed_1) {
    auto e = NDArrayFactory::create<float>('c', {1, 5, 5, 6}, {0.55744928f, 0.76827729f, 1.09401524f, 0.00000000f, 0.00000000f, 0.00000000f, 0.56373537f, 0.90029907f, 0.78997850f, 0.00000000f, 0.00000000f, 0.00000000f, 0.14252824f, 0.95961076f, 0.87750554f, 0.00000000f, 0.00000000f, 0.00000000f, 0.44874173f, 0.99537718f, 1.17154264f, 0.00000000f, 0.00000000f, 0.00000000f, 0.60377145f, 0.79939061f, 0.56031001f, 0.00000000f, 0.00000000f, 0.00000000f, 0.52975273f, 0.90678585f, 0.73763013f, 0.00000000f, 0.00000000f, 0.00000000f, 0.22146404f, 0.82499605f, 0.47222072f, 0.00000000f, 0.00000000f, 0.00000000f, 0.42772964f, 0.39793295f, 0.71436501f, 0.00000000f, 0.00000000f, 0.00000000f, 0.48836520f, 1.01658893f, 0.74419701f, 0.00000000f, 0.00000000f, 0.00000000f, 0.78984612f, 0.94083673f, 0.83841157f, 0.00000000f, 0.00000000f, 0.00000000f, 0.40448499f, 0.67732805f, 0.75499672f, 0.00000000f, 0.00000000f, 0.00000000f, 0.43675962f, 0.79476535f, 0.72976631f, 0.00000000f, 0.00000000f, 0.00000000f, 0.58808053f, 0.65222591f, 0.72552216f, 0.00000000f, 0.00000000f, 0.00000000f, 0.37445742f, 1.22581339f, 1.05341125f, 0.00000000f, 0.00000000f, 0.00000000f, 0.30095795f, 0.59941679f, 0.63323414f, 0.00000000f, 0.00000000f, 0.00000000f, 0.24199286f, 1.02546394f, 0.69537812f, 0.00000000f, 0.00000000f, 0.00000000f, 0.23628944f, 0.90791851f, 1.01209974f, 0.00000000f, 0.00000000f, 0.00000000f, 0.62740159f, 0.56518674f, 0.76692569f, 0.00000000f, 0.00000000f, 0.00000000f, 0.13327584f, 0.32628393f, 0.10280430f, 0.00000000f, 0.00000000f, 0.00000000f, 0.42691272f, 0.25625113f, 0.30524066f, 0.00000000f, 0.00000000f, 0.00000000f, 0.17797673f, 0.84179950f, 0.80061519f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00199084f, 0.51838887f, 0.43932241f, 0.00000000f, 0.00000000f, 0.00000000f, 0.16684581f, 0.50822425f, 0.48668745f, 0.00000000f, 0.00000000f, 0.00000000f, 0.16749343f, 0.93093169f, 0.86871749f, 0.00000000f, 0.00000000f, 0.00000000f, 0.17486368f, 0.44460732f, 0.44499981f, 0.00000000f, 0.00000000f, 0.00000000f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/channels_last_b1_k2_s1_d1_SAME_crelu.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(9));

    auto z = graph->getVariableSpace()->getVariable(9)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    z->printIndexedBuffer("z");

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_tensor_array_1) {
    auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.77878559f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f, 0.18039072f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/tensor_array_close_sz1_float32_nodynamic_noname_noshape.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(5));

    auto z = graph->getVariableSpace()->getVariable(5)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_tensor_array_2) {
    auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.77878559f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f, 0.18039072f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/tensor_array_split_sz1_float32_nodynamic_noname_noshape.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(6));

    auto z = graph->getVariableSpace()->getVariable(6)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_tensor_array_3) {
    auto e = NDArrayFactory::create<int>('c', {3, 2, 3}, {7, 2, 9, 4, 3, 3, 8, 7, 0, 0, 6, 8, 7, 9, 0, 1, 1, 4});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/tensor_array_stack_sz3-1_int32_dynamic_name_shape.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();


    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(15));

    auto z = graph->getVariableSpace()->getVariable(15)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_tensor_array_4) {
    auto e = NDArrayFactory::create<Nd4jLong>('c', {2, 3}, {4, 3, 1, 1, 1, 0});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/tensor_array_unstack_sz1_int64_nodynamic_noname_shape2-3.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();


    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(11));

    auto z = graph->getVariableSpace()->getVariable(11)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_assert_4) {
    auto e = NDArrayFactory::create<Nd4jLong>('c', {2, 2}, {1, 1, 1, 1});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/assert_type_rank2_int64.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();


    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1));

    auto z = graph->getVariableSpace()->getVariable(1)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

// TEST_F(OneOffTests, test_cond_true_1) {
//     auto e = NDArrayFactory::create<float>('c', {5}, {1.f, 2.f, 3.f, 4.f, 5.f});

//     auto graph = GraphExecutioner::importFromFlatBuffers("./resources/cond_true.fb");
//     ASSERT_TRUE(graph != nullptr);

//     graph->printOut();


//     Nd4jStatus status = GraphExecutioner::execute(graph);
//     ASSERT_EQ(Status::OK(), status);
//     ASSERT_TRUE(graph->getVariableSpace()->hasVariable(6));

//     auto z = graph->getVariableSpace()->getVariable(6)->getNDArray();
//     ASSERT_TRUE(z != nullptr);

//     z->printIndexedBuffer("z buffer");

//     ASSERT_EQ(e, *z);

//     delete graph;
// }

/*
TEST_F(OneOffTests, test_cond_false_1) {
    auto e = NDArrayFactory::create<float>('c', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/cond_false.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();


    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(6));

    auto z = graph->getVariableSpace()->getVariable(6)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    z->printIndexedBuffer("z buffer");

    ASSERT_EQ(e, *z);

    delete graph;
}
*/

TEST_F(OneOffTests, test_identity_n_2) {
    auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.77878559f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f, 0.18039072f});

    nd4j::ops::identity_n op;

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/identity_n_2.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();


    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1));
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1, 1));

    auto z = graph->getVariableSpace()->getVariable(1)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

TEST_F(OneOffTests, test_non2d_1) {
    auto e = NDArrayFactory::create<float>('c', {1, 1}, {5.42746449f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/non2d_1.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(3));

    auto z = graph->getVariableSpace()->getVariable(3)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);


    delete graph;
}

TEST_F(OneOffTests, test_reduce_all_1) {
    auto e = NDArrayFactory::create<bool>('c', {1, 4}, {true, false, false, false});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/reduce_all_rank2_d0_keep.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1));

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(2));
    auto in = graph->getVariableSpace()->getVariable(2)->getNDArray();


    auto z = graph->getVariableSpace()->getVariable(1)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);


    delete graph;
}

TEST_F(OneOffTests, test_lrn_1) {
    auto e = NDArrayFactory::create<float>('c', {2, 4, 4, 8}, {0.72673398f, 0.72244084f, 0.65190971f, 0.20479436f, 0.63841927f, 0.16429311f, 0.47735623f, 0.84601802f, 0.51100832f, 0.83933938f, 0.07637496f, 0.70744967f, 0.59434658f, 0.53122658f, 0.15273477f, 0.31313354f, 0.14679013f, 0.43116447f, 0.91072536f, 0.54495502f, 0.60216504f, 0.16226815f, 0.65245372f, 0.22468951f, 0.05115855f, 0.42691258f, 0.73630011f, 0.20842661f, 0.85568416f, 0.74199051f, 0.23918203f, 0.38905996f, 0.65262794f, 0.70963490f, 0.49649292f, 0.66844928f, 0.25772896f, 0.00624168f, 0.03414882f, 0.07292996f, 0.79563236f, 0.70111781f, 0.06900901f, 0.55988204f, 0.32691985f, 0.56760389f, 0.56170380f, 0.18566820f, 0.61561924f, 0.87468338f, 0.27482313f, 0.31954449f, 0.15619321f, 0.81941688f, 0.53810787f, 0.07410496f, 0.86357057f, 0.62125850f, 0.38212773f, 0.21973746f, 0.39605364f, 0.50292832f, 0.67988282f, 0.48182553f, 0.21579312f, 0.92945135f, 0.13491257f, 0.14810418f, 0.51570767f, 0.01417551f, 0.44059932f, 0.15463555f, 0.43939117f, 0.25176623f, 0.33246401f, 0.27868715f, 0.62812716f, 0.36218232f, 0.83123064f, 0.18322028f, 0.69190371f, 0.64964098f, 0.59286875f, 0.63276929f, 0.51176143f, 0.24714552f, 0.71379036f, 0.04463276f, 0.30318832f, 0.46065491f, 0.74381822f, 0.08990894f, 0.70277166f, 0.46120697f, 0.08017386f, 0.79538876f, 0.57498217f, 0.42726749f, 0.54888463f, 0.77145094f, 0.42133608f, 0.62393188f, 0.26213539f, 0.20285931f, 0.40793586f, 0.76919287f, 0.32619563f, 0.46802196f, 0.10585602f, 0.77520430f, 0.17103595f, 0.81063843f, 0.49952045f, 0.43753168f, 0.49716589f, 0.62564862f, 0.24522111f, 0.22645080f, 0.67360777f, 0.27542791f, 0.54475069f, 0.47039139f, 0.61613292f, 0.11884533f, 0.25952721f, 0.07941547f, 0.50383574f, 0.17706800f, 0.22063354f, 0.39298156f, 0.05262897f, 0.35281670f, 0.58542049f, 0.08724644f, 0.44015333f, 0.06217420f, 0.55999058f, 0.49639198f, 0.24356756f, 0.16928019f, 0.07262222f, 0.88610303f, 0.36227691f, 0.30532160f, 0.46558958f, 0.68724936f, 0.43359837f, 0.47884533f, 0.15492828f, 0.31899589f, 0.71925861f, 0.89685601f, 0.21630493f, 0.20829079f, 0.62069362f, 0.27684751f, 0.85515481f, 0.16900150f, 0.88375843f, 0.16574328f, 0.90707570f, 0.42697936f, 0.57044196f, 0.28660366f, 0.89634472f, 0.53366786f, 0.53777111f, 0.02245709f, 0.28782824f, 0.51147747f, 0.53804207f, 0.63065857f, 0.16481975f, 0.77783436f, 0.83846682f, 0.81526840f, 0.05372881f, 0.30025366f, 0.00382434f, 0.74619126f, 0.53056949f, 0.36895069f, 0.36868665f, 0.36383235f, 0.28313419f, 0.76751631f, 0.43787196f, 0.40706897f, 0.60503763f, 0.57529414f, 0.00334400f, 0.48582658f, 0.61580127f, 0.67637223f, 0.03531667f, 0.56674093f, 0.03619293f, 0.57728928f, 0.04089037f, 0.71581155f, 0.88595301f, 0.82495558f, 0.39791623f, 0.47771215f, 0.82810479f, 0.34763816f, 0.05219861f, 0.50241560f, 0.34942892f, 0.63052243f, 0.11668805f, 0.49408942f, 0.53590798f, 0.35757038f, 0.50228578f, 0.33905780f, 0.61803931f, 0.44139981f, 0.38804284f, 0.53102624f, 0.58827806f, 0.67419237f, 0.07855739f, 0.16120273f, 0.25186202f, 0.85384846f, 0.61189389f, 0.49337071f, 0.65005291f, 0.75512350f, 0.00436557f, 0.19363582f, 0.61603606f, 0.38213563f, 0.62929642f, 0.48036635f, 0.58100700f, 0.79667836f, 0.67103785f, 0.74890578f, 0.39760122f, 0.32866973f, 0.48477241f, 0.37567708f, 0.09772851f, 0.07863729f, 0.12172586f, 0.17331666f, 0.51744813f, 0.85402083f, 0.84792399f, 0.07443056f, 0.08461216f, 0.03309918f, 0.29576823f, 0.15721567f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/lrn_dr3_b05_a05_b02.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1));

    auto z = graph->getVariableSpace()->getVariable(1)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    ASSERT_EQ(e, *z);

    delete graph;
}

