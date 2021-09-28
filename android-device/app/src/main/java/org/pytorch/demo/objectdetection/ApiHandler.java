package org.pytorch.demo.objectdetection;


import java.util.ArrayList;

public class ApiHandler {
    ArrayList<Result> lastResults = new ArrayList<>();;

    public void postSplitTensor(QuantizedTensor qx){
        lastResults.clear(); // Clear Last Results
        new AsyncPostTensor(lastResults, "http://192.168.15.172:5000/split").execute(qx);
    }
    public void postImageTensor(QuantizedTensor qx){
        lastResults.clear(); // Clear Last Results
        new AsyncPostTensor(lastResults, "http://192.168.15.172:5000/compute").execute(qx);
    }
}
