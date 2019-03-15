package org.nd4j.deallocator;

import java.lang.ref.WeakReference;

public interface ReferenceTracking<T> {

    WeakReference<T> getNextReference() throws InterruptedException;

    void handleReference(WeakReference<T> holder);

    void handleNullReference();
}
