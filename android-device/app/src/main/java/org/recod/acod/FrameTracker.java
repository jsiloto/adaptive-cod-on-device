package org.recod.acod;


public class FrameTracker {
    private int responseCounter = 0;
    private int frameCounter = 0;
    private long trackerStart;
    private long frameStart = 0;
    private long dnnStart = 0;
    private long requestStart = 0;
    private double averageFrameTime = 0;
    private double averagePreprocessTime = 0;
    private double averageDnnTime = 0;
    private double averageRoundTripTime = 0;


    public FrameTracker() {
        trackerStart = System.currentTimeMillis();
    }

    // All the following methods should be called in this order
    public void RegisterNewFrame() {
        frameCounter++;
        frameStart = System.currentTimeMillis();
    }

    public void RegisterDnnStart() {
        dnnStart = System.currentTimeMillis();
        double tDelta = (dnnStart - frameStart) / 1000.0;
        averagePreprocessTime = CalcAverage(tDelta, averagePreprocessTime, frameCounter);
    }

    public void RegisterRequest() {
        requestStart = System.currentTimeMillis();
        double tDelta = (requestStart - dnnStart) / 1000.0;
        averageDnnTime = CalcAverage(tDelta, averageDnnTime, frameCounter);
    }

    public void RegisterFrameEnd() {
        double tDelta;
        responseCounter++;
        long tCurrent = System.currentTimeMillis();

        tDelta = (tCurrent - trackerStart) / 1000.0;
        averageFrameTime = tDelta / responseCounter;

        tDelta = (tCurrent - requestStart) / 1000.0;
        averageRoundTripTime = CalcAverage(tDelta, averageRoundTripTime, responseCounter);
    }

    public Boolean isDone(){
        return frameCounter == responseCounter;
    }

    public String statistics() {
        String message = String.format("Processed Frames:%d\n", frameCounter) +
                String.format("Resolved Frames:%d\n", responseCounter) +
                String.format("Average FPS:%2.3f\n", 1 / averageFrameTime) +
                String.format("Average Encoder Processing Time:%2.3f\n", averageDnnTime) +
                String.format("Average Round Trip Time:%2.3f", averageRoundTripTime);

        return message;
    }

    private static double CalcAverage(double delta, double prevAverage, int counter) {
        double avg = (prevAverage * (counter - 1) + delta) / counter;
        return avg;
    }
}
