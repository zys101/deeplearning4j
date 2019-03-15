package org.nd4j.deallocator;

public class ReferenceTrackingFactory<T> {

    public static ReferenceTracking create() {
        return new DefaultReferenceTracker<>();
    }
}
