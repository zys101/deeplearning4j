package org.deeplearning4j.models.node2vec;

import org.deeplearning4j.models.sequencevectors.graph.enums.SamplingMode;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.graph.walkers.impl.NearestVertexWalker;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class Node2VecTest {
    private static long seed = 119;

    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testPVDM1() throws Exception {
        Graph<VocabWord, Integer> graph = getGraph();
        GraphWalker<VocabWord> walker = new NearestVertexWalker.Builder<VocabWord>(graph)
                .setSamplingMode(SamplingMode.MAX_POPULARITY)
                .setDepth(0)
                .build();

        Node2Vec<VocabWord, Integer> node2Vec = new Node2Vec.Builder<VocabWord, Integer>(walker).build();
    }

    /**
     * This method returns random generated
     *
     * @return
     */
    protected static Graph<VocabWord, Integer> getGraph() {
        List<Vertex<VocabWord>> list = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            list.add(new Vertex<VocabWord>(i, new VocabWord(1.0, String.valueOf(i))));
        }

        Graph<VocabWord, Integer> graph = new Graph<>(list);
        Random rng = new Random(seed);

        int numConns = 0;

        for (int i = 0; i < 10000; i++) {
            int conns = rng.nextInt( 10) + 5;
            if (i == 0)
                numConns = conns;

            for (int e = 0; e < conns; e++) {
                graph.addEdge(i, getConnection(i, list.size(), rng), 1, true);
            }
        }





        return graph;
    }

    protected static int getConnection(int  currentIndex, int limit, Random rng) {
        return rng.nextInt(limit);
    }
}