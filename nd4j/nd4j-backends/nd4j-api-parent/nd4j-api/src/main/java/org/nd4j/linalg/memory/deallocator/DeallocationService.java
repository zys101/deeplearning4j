package org.nd4j.linalg.memory.deallocator;

import org.nd4j.linalg.executors.ExecutorServiceProvider;

import java.util.concurrent.ExecutorService;

public class DeallocationService {

    private static DeallocationService INSTANCE = new DeallocationService();

    public static DeallocationService getInstance() {
        return INSTANCE;
    }

    private ExecutorService executorService = ExecutorServiceProvider.getExecutorService();

    public void submit(DeallocatorThread runnable) {
        executorService.submit(runnable);
    }

}
