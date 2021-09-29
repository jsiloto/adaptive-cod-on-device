package org.pytorch.demo.objectdetection;

import java.io.IOException;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class MapAPI {


    public static Boolean clearServerState(String url){
        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder()
                .url(url)
                .delete()
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()){
                return false;
            }
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        return true;
    }

    public static Boolean getServerState(String url){
        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder()
                .url(url)
                .get()
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()){
                return false;
            }
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        return true;
    }


}
