package org.recod.acod;

import android.graphics.Rect;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.util.ArrayList;

import okhttp3.Headers;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class PostTensor {
    private OkHttpClient client;
    private Response response;
    private String url;
    private Gson jsonParser = new Gson();
    private FrameTracker frameTracker;
    private AsyncPostTensor.onPostExecuteCallback callback;

    public PostTensor(String url){
        this.url = url;
        client = new OkHttpClient();
    }

    public ArrayList<Result> post(QuantizedTensor... qxs){
        QuantizedTensor qx = qxs[0];
        Headers requestHeader = new Headers.Builder()
                .add("w", Integer.toString(qx.originalWidth))
                .add("h", Integer.toString(qx.originalHeight))
                .add("ct", Integer.toString(qx.c))
                .add("ht", Integer.toString(qx.h))
                .add("wt", Integer.toString(qx.w))
                .add("mode", qx.mode)
                .add("image_id", qx.imageId)
                .add("zero_point", Integer.toString(qx.zero_point))
                .add("scale", Float.toString(qx.scale))
                .build();
        RequestBody requestBody = RequestBody.create(qx.qx);

        Request request = new Request.Builder()
                .url(url)
                .headers(requestHeader)
                .post(requestBody)
                .build();

        ArrayList<Result> results = new ArrayList<>();;
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()){
                results = parseResponse(response);
            }
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        return results;
    }

    private ArrayList<Result> parseResponse(Response response) throws IOException {
        float threshold = 0.5f;

        ArrayList<Result> results = new ArrayList<>();
        final String responseBody = response.body().string();
        JsonObject jsonObject =  JsonParser.parseString(responseBody).getAsJsonObject();
        JsonArray data = jsonObject.getAsJsonArray("data");
        for(JsonElement obj: data){
            Result r = parseJSONResult(obj.getAsJsonObject());
            if (r.score > threshold){
                results.add(r);
            }
        }
        return results;
    }
    private Result parseJSONResult(JsonObject obj){
        int[] numbers = jsonParser.fromJson(obj.getAsJsonObject().get("bbox"), int[].class);

        int x = numbers[0];
        int y = numbers[1];
        int w = numbers[2];
        int h = numbers[3];
        return new Result(
                obj.get("class").getAsInt()-1,
                obj.get("score").getAsFloat(),
                new Rect(x, y, x+w, y+h));
    }


}
