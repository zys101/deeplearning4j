package org.nd4j.linalg.memory.deallocator;

import lombok.NonNull;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.Map;

public class DeallocatorThread<T extends ReferenceTracking> extends Thread implements Runnable {

    private final ReferenceQueue<T> queue;
    private Map<String, Nd4jWorkspace.GarbageWorkspaceReference> referenceMap;

    public DeallocatorThread(int threadId, @NonNull ReferenceQueue<T> queue,
                             Map<String, Nd4jWorkspace.GarbageWorkspaceReference> referenceMap) {

        this.queue = queue;
        this.referenceMap = referenceMap;
        this.setName("DeallocatorThread" + threadId);
        this.setDaemon(true);
    }

    @Override
    public void run() {

        while (true) {
            try {
                WeakReference<T> ref = (WeakReference<T>)queue.remove();
                Nd4jWorkspace.GarbageWorkspaceReference decoratedRef = referenceMap.get(ref.get());
                if (decoratedRef != null) {
                    decoratedRef.deallocatorCallback.handleReference(ref);
                } else {
                    decoratedRef.deallocatorCallback.handleNullReference();
                }
            } /*catch (InterruptedException e) {
                // do nothing
            }*/ catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
