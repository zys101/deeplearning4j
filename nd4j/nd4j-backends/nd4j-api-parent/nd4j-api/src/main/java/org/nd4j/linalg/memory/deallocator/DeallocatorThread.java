package org.nd4j.linalg.memory.deallocator;

import lombok.NonNull;

import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.Map;

public class DeallocatorThread<T extends ReferenceTracking> extends Thread implements Runnable {

    private final ReferenceQueue<T> queue;
    //private final Map<Long, GarbageStateReference> referenceMap;
    //private final ReferenceTracking<T> tracker;

    /*protected DeallocatorThread(int threadId, @NonNull ReferenceQueue<T> queue,
                                Map<Long, GarbageStateReference> referenceMap) {

        this.tracker = ReferenceTrackingFactory.create();
        this.queue = queue;
        this.referenceMap = referenceMap;
        this.setName("DeallocatorThread" + threadId);
        this.setDaemon(true);
    }*/

    public DeallocatorThread(int threadId, @NonNull ReferenceQueue<T> queue) {

        this.queue = queue;
        //this.referenceMap = null;
        this.setName("DeallocatorThread" + threadId);
        this.setDaemon(true);
    }

    @Override
    public void run() {

        while (true) {
            try {
                Reference<T> ref = (Reference<T>) queue.remove();
                if (ref != null) {
                    T reference = ref.get();

                    if (reference != null) {
                        reference.handleReference();
                    } else {
                        reference.handleNullReference();
                    }
                }
            } /*catch (InterruptedException e) {
                // do nothing
            }*/ catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
