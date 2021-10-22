package org.pytorch.demo.objectdetection;


import java.util.ArrayList;
import java.util.concurrent.ExecutionException;

public class ApiHandler {
    ArrayList<Result> lastResults = new ArrayList<>();
    String url = "http://192.168.1.23:5000/";


    public void postSplitTensor(QuantizedTensor qx) throws ExecutionException, InterruptedException {
        new AsyncPostTensor(lastResults, url + "split").execute(qx).get();
//        new AsyncPostTensor(lastResults, url+"split").execute(qx);
    }

    public void postImageTensor(QuantizedTensor qx) throws ExecutionException, InterruptedException {
        new AsyncPostTensor(lastResults, url + "compute").execute(qx).get();
    }

    public void clearServerMAP() {
        MapAPI.clearServerState(url + "map");
    }

    public void getServerMAP() {
        MapAPI.getServerState(url + "map");
    }

}
