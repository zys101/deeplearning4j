package org.nd4j.linalg.memory.deallocator;

import org.nd4j.linalg.executors.ExecutorServiceProvider;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import java.lang.ref.ReferenceQueue;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;

public class DeallocationService {

    private static final int THRESHOLD = 4;
    private Map<String, Nd4jWorkspace.DeallocatableWorkspaceReference> referenceMap = new ConcurrentHashMap<>();

    DeallocationService() {
        for (int i = 0; i < THRESHOLD; ++i) {

            ReferenceQueue queue = new ReferenceQueue();
            DeallocatorThread thread = new DeallocatorThread(i, queue, referenceMap);
            executorService.submit(thread);
        }
    }

    private static DeallocationService INSTANCE = null;

    public static synchronized DeallocationService getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new DeallocationService();
        }
        return INSTANCE;
    }

    private ExecutorService executorService = ExecutorServiceProvider.getExecutorService();

    public void register(Map<String, Nd4jWorkspace.DeallocatableWorkspaceReference> referenceMap) {
        this.referenceMap = referenceMap;
    }
}
