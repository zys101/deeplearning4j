package org.nd4j.linalg.memory.deallocator;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.executors.ExecutorServiceProvider;

import java.lang.ref.ReferenceQueue;
import java.util.concurrent.ExecutorService;

public class DeallocationService<T> {

    private static DeallocationService INSTANCE = new DeallocationService();

    public static DeallocationService getInstance() {
        return INSTANCE;
    }

    private ExecutorService executorService = ExecutorServiceProvider.getExecutorService();

    public void submit(ReferenceQueue<T> queue, ReferenceTracking<T> tracker) {
        DeallocatorThread<T> deallocationThread =
                new DeallocatorThread<>(0, queue, tracker);
        executorService.submit(deallocationThread);
    }

}
