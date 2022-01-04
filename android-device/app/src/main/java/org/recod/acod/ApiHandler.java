package org.recod.acod;


import java.io.IOException;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class ApiHandler {
    String url = "http://192.168.1.102:5000/";

    ApiHandler() {
    }

    ApiHandler(String url) {
        this.url = url;
    }

    public void postSplitTensor(QuantizedTensor qx, AsyncPostTensor.onPostExecuteCallback callback) {
        new AsyncPostTensor(url + "split", callback).execute(qx);
    }

    public void clearServerMAP() {
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
