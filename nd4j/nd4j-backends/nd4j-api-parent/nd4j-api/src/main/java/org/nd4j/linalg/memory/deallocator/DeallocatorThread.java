package org.nd4j.linalg.memory.deallocator;

import lombok.NonNull;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.Map;

public class DeallocatorThread<T> extends Thread implements Runnable {

    private final ReferenceQueue<T> queue;
    //private final Map<Long, GarbageStateReference> referenceMap;
    private final ReferenceTracking<T> tracker;

    /*protected DeallocatorThread(int threadId, @NonNull ReferenceQueue<T> queue,
                                Map<Long, GarbageStateReference> referenceMap) {

        this.tracker = ReferenceTrackingFactory.create();
        this.queue = queue;
        this.referenceMap = referenceMap;
        this.setName("DeallocatorThread" + threadId);
        this.setDaemon(true);
    }*/

    public DeallocatorThread(int threadId, @NonNull ReferenceQueue<T> queue, ReferenceTracking<T> tracker) {

        this.tracker = tracker;
        this.queue = queue;
        //this.referenceMap = null;
        this.setName("DeallocatorThread" + threadId);
        this.setDaemon(true);
    }

    @Override
    public void run() {

        while (true) {
            try {
                WeakReference<T> reference = tracker.getNextReference();

                if (reference != null) {
                    tracker.handleReference(reference);
                } else {
                    tracker.handleNullReference();
                }
            } catch (InterruptedException e) {
                // do nothing
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
