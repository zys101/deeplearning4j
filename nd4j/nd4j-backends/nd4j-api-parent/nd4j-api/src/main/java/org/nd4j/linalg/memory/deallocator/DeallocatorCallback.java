package org.nd4j.linalg.memory.deallocator;

import java.lang.ref.WeakReference;

public interface DeallocatorCallback {

    WeakReference<Object> getNextReference() throws InterruptedException;

    void deallocate(WeakReference<Object> reference);

    void handleNullReference();
}
