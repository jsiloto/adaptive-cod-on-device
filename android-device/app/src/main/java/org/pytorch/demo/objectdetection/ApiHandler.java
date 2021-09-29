package org.pytorch.demo.objectdetection;


import java.util.ArrayList;
import java.util.concurrent.ExecutionException;

public class ApiHandler {
    ArrayList<Result> lastResults = new ArrayList<>();

    public void postSplitTensor(QuantizedTensor qx) throws ExecutionException, InterruptedException {
        new AsyncPostTensor(lastResults, "http://192.168.15.172:5000/split").execute(qx).get();;
    }
    public void postImageTensor(QuantizedTensor qx) throws ExecutionException, InterruptedException {
        new AsyncPostTensor(lastResults, "http://192.168.15.172:5000/compute").execute(qx).get();;
    }
}
