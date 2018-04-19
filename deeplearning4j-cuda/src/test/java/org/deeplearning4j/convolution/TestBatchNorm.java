package org.deeplearning4j.convolution;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.normalization.CudnnBatchNormalizationHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Field;

import static org.junit.Assert.*;

public class TestBatchNorm {

    @Test
    public void testCompareMLN() throws Exception {
//        Nd4j.setDataType(DataBuffer.Type.FLOAT);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(new ConvolutionLayer.Builder().nIn(3).nOut(3).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ConvolutionLayer.Builder().nIn(3).nOut(3).build())
                .layer(new OutputLayer.Builder().nOut(10).build())
                .setInputType(InputType.convolutional(10, 10, 3))
                .build();

        MultiLayerNetwork net1 = new MultiLayerNetwork(conf);
        net1.init();

        MultiLayerNetwork net2 = new MultiLayerNetwork(conf.clone());
        net2.init();

        INDArray p1 = net1.params();
        p1.assign(Nd4j.rand(1, p1.length()));

        net2.params().assign(p1);

        Field f = org.deeplearning4j.nn.layers.normalization.BatchNormalization.class.getDeclaredField("helper");
        f.setAccessible(true);

        assertNotNull(f.get(net1.getLayer(1)));
        assertTrue(f.get(net1.getLayer(1)) instanceof CudnnBatchNormalizationHelper);

        f.set(net2.getLayer(1), null);
        assertNull(f.get(net2.getLayer(1)));

        INDArray in = Nd4j.rand(new int[]{2, 3, 10, 10});
        INDArray out1Train = net1.output(in, true);
        INDArray out2Train = net2.output(in, true);

        INDArray out1Test = net1.output(in, false);
        INDArray out2Test = net1.output(in, false);

        assertEquals(out1Train, out2Train);
        assertEquals(out1Test, out2Test);
    }


    @Test
    public void testCompareCG() throws Exception {
//        Nd4j.setDataType(DataBuffer.Type.FLOAT);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder()
                .addInputs("in")
                .layer("0", new ConvolutionLayer.Builder().nIn(3).nOut(3).build(), "in")
                .layer("1", new BatchNormalization.Builder().build(), "0")
                .layer("2", new ConvolutionLayer.Builder().nIn(3).nOut(3).build(), "1")
                .layer("3", new OutputLayer.Builder().nOut(10).build(), "2")
                .setOutputs("3")
                .setInputTypes(InputType.convolutional(10, 10, 3))
                .build();

        ComputationGraph net1 = new ComputationGraph(conf);
        net1.init();

        ComputationGraph net2 = new ComputationGraph(conf.clone());
        net2.init();

        INDArray p1 = net1.params();
        p1.assign(Nd4j.rand(1, p1.length()));

        net2.params().assign(p1);

        Field f = org.deeplearning4j.nn.layers.normalization.BatchNormalization.class.getDeclaredField("helper");
        f.setAccessible(true);

        assertNotNull(f.get(net1.getLayer(1)));
        assertTrue(f.get(net1.getLayer(1)) instanceof CudnnBatchNormalizationHelper);

        f.set(net2.getLayer(1), null);
        assertNull(f.get(net2.getLayer(1)));

        INDArray in = Nd4j.rand(new int[]{2, 3, 10, 10});
        INDArray out1Train = net1.output(true, in)[0];
        INDArray out2Train = net2.output(true, in)[0];

        INDArray out1Test = net1.output(false, in)[0];
        INDArray out2Test = net1.output(false, in)[0];

        assertEquals(out1Train, out2Train);
        assertEquals(out1Test, out2Test);
    }

}
