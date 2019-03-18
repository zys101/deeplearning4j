package org.nd4j.linalg.memory.deallocator;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.executors.ExecutorServiceProvider;

import java.lang.ref.ReferenceQueue;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

public class DeallocationService<T> {

    private static final int THRESHOLD = 4;
    private DeallocatorThread<T> threads[] = new DeallocatorThread[4];
    private AtomicInteger next = new AtomicInteger(0);

    private static DeallocationService INSTANCE = new DeallocationService();

    public static DeallocationService getInstance() {
        return INSTANCE;
    }

    private ExecutorService executorService = ExecutorServiceProvider.getExecutorService();

    public void submit(ReferenceQueue<T> queue, ReferenceTracking<T> tracker) {

        if (next.incrementAndGet() >= THRESHOLD)
            throw new IllegalStateException("No deallocator threads available");

        threads[next.incrementAndGet()] =
                new DeallocatorThread<>(0, queue, tracker);
        executorService.submit(threads[next.incrementAndGet()]);
        next.incrementAndGet();
    }

}
