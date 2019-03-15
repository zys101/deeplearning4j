package org.nd4j.deallocator;

import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.rng.deallocator.GarbageStateReference;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.LockSupport;

public class DefaultReferenceTracker<T> implements ReferenceTracking<T> {

    private ReferenceQueue<T> queue;
    private Map<Long, GarbageStateReference> referenceMap;

    public void DefaultReferenceTracker() {
        this.queue = new ReferenceQueue<>();
        this.referenceMap = new ConcurrentHashMap<>();
    }

    @Override
    public WeakReference<T> getNextReference() throws InterruptedException {
        GarbageStateReference reference = (GarbageStateReference) queue.remove();
        return (WeakReference<T>) reference;
    }

    @Override
    public void handleReference(WeakReference<T> holder) {
        GarbageStateReference reference = (GarbageStateReference)holder;
        if (reference.getStatePointer() != null) {
            referenceMap.remove(reference.getStatePointer().address());
            NativeOpsHolder.getInstance().getDeviceNativeOps()
                    .destroyRandom(reference.getStatePointer());
        }
    }

    @Override
    public void handleNullReference() {
        LockSupport.parkNanos(5000L);
    }
}
