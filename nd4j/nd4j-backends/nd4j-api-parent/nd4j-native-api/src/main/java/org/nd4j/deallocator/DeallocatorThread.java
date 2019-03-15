package org.nd4j.deallocator;

import lombok.NonNull;
import org.nd4j.rng.deallocator.GarbageStateReference;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.Map;

public class DeallocatorThread<T> extends Thread implements Runnable {

    private final ReferenceQueue<T> queue;
    private final Map<Long, GarbageStateReference> referenceMap;
    private final ReferenceTracking<T> tracker;

    protected DeallocatorThread(int threadId, @NonNull ReferenceQueue<T> queue,
                                Map<Long, GarbageStateReference> referenceMap) {

        this.tracker = ReferenceTrackingFactory.create();
        this.queue = queue;
        this.referenceMap = referenceMap;
        this.setName("DeallocatorThread" + threadId);
        this.setDaemon(true);
    }

    protected DeallocatorThread(int threadId, @NonNull ReferenceQueue<T> queue, ReferenceTracking<T> tracker) {

        this.tracker = tracker;
        this.queue = queue;
        this.referenceMap = null;
        this.setName("DeallocatorThread" + threadId);
        this.setDaemon(true);
    }

    @Override
    public void run() {

        while (true) {
            try {
                WeakReference<T> reference = tracker.getNextReference();

                if (reference != null) {
                    tracker.cleanReferencee(reference);
                } else {
                    tracker.handleNullReferencee();
                }
            } catch (InterruptedException e) {
                // do nothing
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
