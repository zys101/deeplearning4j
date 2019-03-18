package org.nd4j.linalg.memory.deallocator;

import lombok.NonNull;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.Map;

public class DeallocatorThread extends Thread implements Runnable {

    private final ReferenceQueue<Object> queue;
    private Map<String, Nd4jWorkspace.DeallocatableWorkspaceReference> referenceMap;

    public DeallocatorThread(int threadId, @NonNull ReferenceQueue<Object> queue,
                             Map<String, Nd4jWorkspace.DeallocatableWorkspaceReference> referenceMap) {

        this.queue = queue;
        this.referenceMap = referenceMap;
        this.setName("DeallocatorThread" + threadId);
        this.setDaemon(true);
    }

    @Override
    public void run() {

        while (true) {
            try {
                WeakReference ref = (WeakReference) queue.remove();
                Nd4jWorkspace.DeallocatableWorkspaceReference decoratedRef = referenceMap.get(ref.get());
                if (decoratedRef != null) {
                    decoratedRef.callbackService.deallocate(ref);
                } else {
                    decoratedRef.callbackService.handleNullReference();
                }
            } /*catch (InterruptedException e) {
                // do nothing
            }*/ catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
