package org.recod.acod;


import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class ApiHandler {
    ArrayList<Result> lastResults = new ArrayList<>();
    String url = "http://192.168.15.172:5000/";
    int[] response_counter;

    public ApiHandler() {
        response_counter = new int[1];
        response_counter[0] = 0;
    }

    public void postSplitTensor(QuantizedTensor qx) throws ExecutionException, InterruptedException {
        new AsyncPostTensor(lastResults, url + "split", response_counter).execute(qx);
    }

    public void clearServerMAP() {
        response_counter[0] = 0;
        MapAPI.clearServerState(url + "map");
    }

    public String getServerMAP() {
        String result = MapAPI.getServerState(url + "map");
        return result;
    }

    public Boolean postData(String data, String endpoint) throws IOException {
        OkHttpClient client = new OkHttpClient();
        RequestBody requestBody = RequestBody.create(data.getBytes());
        Request request = new Request.Builder()
                .url(url + "data/" + endpoint)
                .post(requestBody)
                .build();

        Response response = client.newCall(request).execute();
        if (!response.isSuccessful()) {
            return false;
        } else {
            return true;
        }
    }

}