package org.deeplearning4j.convolution;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.convolution.CudnnConvolutionHelper;
import org.deeplearning4j.nn.layers.convolution.subsampling.CudnnSubsamplingHelper;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer;
import org.deeplearning4j.nn.layers.normalization.CudnnBatchNormalizationHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Triple;

import java.lang.reflect.Field;

import static org.junit.Assert.*;

public class TestBatchNorm {

    @Test
    public void testCompareMLN() throws Exception {
//        Nd4j.setDataType(DataBuffer.Type.FLOAT);

        for (int minibatch : new int[]{1, 5}) {

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

            Field f2 = org.deeplearning4j.nn.layers.convolution.ConvolutionLayer.class.getDeclaredField("helper");
            f2.setAccessible(true);
            f2.set(net2.getLayer(0), null);
            f2.set(net2.getLayer(2), null);


            INDArray in = Nd4j.rand(new int[]{minibatch, 3, 10, 10});
            INDArray out1Train = net1.output(in, true);
            INDArray out2Train = net2.output(in, true);

            INDArray out1Test = net1.output(in, false);
            INDArray out2Test = net1.output(in, false);

            assertEquals(out1Train, out2Train);
            assertEquals(out1Test, out2Test);


            for (int i = 0; i < 10; i++) {
                INDArray input = Nd4j.rand(new int[]{minibatch, 3, 10, 10});
                INDArray labels = Nd4j.rand(minibatch, 10);
                net1.fit(input, labels);
                net2.fit(input, labels);
            }

            in = Nd4j.rand(new int[]{minibatch, 3, 10, 10});
            out1Train = net1.output(in, true);
            out2Train = net2.output(in, true);

            out1Test = net1.output(in, false);
            out2Test = net1.output(in, false);

            assertEquals(out1Train, out2Train);
            assertEquals(out1Test, out2Test);
        }
    }


    @Test
    public void testCompareCG() throws Exception {
//        Nd4j.setDataType(DataBuffer.Type.FLOAT);

        for (int c = 0; c < 2; c++) {
            for (int minibatch : new int[]{1, 5}) {

                Triple<ComputationGraphConfiguration, int[], int[][]> p = getConf(c);
                ComputationGraphConfiguration conf = p.getFirst();

                int[] inShape = p.getSecond();
                inShape[0] = minibatch;
                int[][] outShapes = p.getThird();
                for (int i = 0; i < outShapes.length; i++) {
                    outShapes[i][0] = minibatch;
                }

                ComputationGraph net1 = new ComputationGraph(conf);
                net1.init();

                ComputationGraph net2 = new ComputationGraph(conf.clone());
                net2.init();

                INDArray p1 = net1.params();
                p1.assign(Nd4j.rand(1, p1.length()));

                net2.params().assign(p1);

                Field f = org.deeplearning4j.nn.layers.convolution.ConvolutionLayer.class.getDeclaredField("helper");
                f.setAccessible(true);

                Field f2 = org.deeplearning4j.nn.layers.normalization.BatchNormalization.class.getDeclaredField("helper");
                f2.setAccessible(true);

                Field f3 = SubsamplingLayer.class.getDeclaredField("helper");
                f2.setAccessible(true);

                for (int i = 0; i < net2.getNumLayers(); i++) {
                    if (net2.getLayer(i) instanceof org.deeplearning4j.nn.layers.convolution.ConvolutionLayer) {
                        assertNotNull(f.get(net1.getLayer(i)));
                        assertTrue(f.get(net1.getLayer(i)) instanceof CudnnConvolutionHelper);

                        f.set(net2.getLayer(i), null);
                    }

                    if (net2.getLayer(i) instanceof org.deeplearning4j.nn.layers.normalization.BatchNormalization) {
                        assertNotNull(f2.get(net1.getLayer(i)));
                        assertTrue(f2.get(net1.getLayer(i)) instanceof CudnnBatchNormalizationHelper);

                        f2.set(net2.getLayer(i), null);
                    }

                    if (net2.getLayer(i) instanceof SubsamplingLayer) {
                        assertNotNull(f3.get(net1.getLayer(i)));
                        assertTrue(f3.get(net1.getLayer(i)) instanceof CudnnSubsamplingHelper);

                        f3.set(net2.getLayer(i), null);
                    }
                }

                INDArray in = Nd4j.rand(inShape);
                INDArray out1Train = net1.output(true, in)[0];
                INDArray out2Train = net2.output(true, in)[0];

                INDArray out1Test = net1.output(false, in)[0];
                INDArray out2Test = net1.output(false, in)[0];

                assertEquals(out1Train, out2Train);
                assertEquals(out1Test, out2Test);

                INDArray input = Nd4j.rand(inShape);
                INDArray[] labels = new INDArray[outShapes.length];
                for (int j = 0; j < labels.length; j++) {
                    labels[j] = Nd4j.rand(outShapes[j]);
                }
                for (int i = 0; i < 10; i++) {
                    net1.fit(new MultiDataSet(new INDArray[]{input}, labels));
                    net2.fit(new MultiDataSet(new INDArray[]{input}, labels));
                }

                out1Train = net1.output(true, in)[0];
                out2Train = net2.output(true, in)[0];

                out1Test = net1.output(false, in)[0];
                out2Test = net1.output(false, in)[0];

                assertEquals(out1Train, out2Train);
                assertEquals(out1Test, out2Test);
            }
        }
    }

    private static Triple<ComputationGraphConfiguration, int[], int[][]> getConf(int idx) {

        if (idx == 0) {
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

            return new Triple<>(conf, new int[]{-1, 3, 10, 10}, new int[][]{{-1, 10}});
        } else if (idx == 1) {
            int height = 28;
            int width = 28;
            int channels = 1; // single channel for grayscale images
            int out1Size = 5;
            int out2Size = 2;
            return new Triple<>(mkResNet(height, width, channels, 3, 8, 16, 1, out1Size, out2Size), new int[]{-1, channels, height, width}, new int[][]{{-1, out1Size}, {-1, out2Size}});
        } else {
            throw new RuntimeException();
        }

    }

    public static final boolean USE_BATCHNORM = true;

    public static String addBlock(ComputationGraphConfiguration.GraphBuilder b, int features, int id, String input) {
        b.addLayer("b" + id + "conv1", new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}).nOut(features).build(), input);

        String tmp = "b" + id + "conv1";
        b.addLayer("b" + id + "bn1", new BatchNormalization(), "b" + id + "conv1");
        tmp = "b" + id + "bn1";

        b.addLayer("b" + id + "act1", new ActivationLayer.Builder().activation(Activation.RELU).build(), tmp);

        b.addLayer("b" + id + "conv2", new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}).nOut(features).build(), "b" + id + "act1");

        tmp = "b" + id + "conv2";
        b.addLayer("b" + id + "bn2", new BatchNormalization(), "b" + id + "conv2");
        tmp = "b" + id + "bn2";

        b.addVertex("short" + id, new ElementWiseVertex(ElementWiseVertex.Op.Add), tmp, input);
        b.addLayer("b" + id + "act2", new ActivationLayer.Builder().activation(Activation.RELU).build(), "short" + id);
        return "b" + id + "act2";
    }

    public static ComputationGraphConfiguration mkResNet(int h, int w, int d, int baseKernelSize, int baseFeatures, int features, int blocks, int out1Size, int out2Size) {
        ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
                .seed(42)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(1e-4, 0.9))
                .weightInit(WeightInit.XAVIER_FAN_IN)
                .miniBatch(true).convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        b.addInputs("input").setInputTypes(InputType.convolutional(h, w, d))
                .addLayer("stem-conv",
                        new ConvolutionLayer.Builder(baseKernelSize, baseKernelSize).stride(1, 1)
                                .nOut(baseFeatures).build(),
                        "input");

        String tmp = "stem-conv";
        tmp = "stem-bn";
        b.addLayer("stem-bn", new BatchNormalization(), "stem-conv");
        b.addLayer("stem-act", new ActivationLayer.Builder().activation(Activation.RELU).build(), tmp);

        String prevLayer = "stem-act";

        if (baseFeatures != features && blocks > 0) {
            prevLayer = "converter";
            b.addLayer("converter",
                    new ConvolutionLayer.Builder(new int[]{1, 1}, new int[]{1, 1}).nOut(features).build(),
                    "stem-act");
        }

        for (int bId = 0; bId < blocks; bId++) {
            prevLayer = addBlock(b, features, bId, prevLayer);
        }

        b.addLayer("moveOut", new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                .nOut(out1Size).activation(Activation.SOFTMAX).build(), prevLayer);
        b.addLayer("winOut", new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                .nOut(out2Size).activation(Activation.SOFTMAX).build(), prevLayer);
        b.setOutputs("moveOut", "winOut");

        b.backprop(true).pretrain(false);

        return b.build();
    }
}
