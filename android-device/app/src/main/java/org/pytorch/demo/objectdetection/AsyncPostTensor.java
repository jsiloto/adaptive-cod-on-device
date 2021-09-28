package org.pytorch.demo.objectdetection;
import android.graphics.Rect;
import android.os.AsyncTask;

import com.google.gson.*;

import java.io.IOException;
import java.util.ArrayList;

import okhttp3.Headers;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class AsyncPostTensor extends AsyncTask<QuantizedTensor, Void, Boolean> {
    private OkHttpClient client;
    private Response response;
    private ArrayList<Result> results;
    private String url;
    private Gson jsonParser = new Gson();

    public AsyncPostTensor(ArrayList<Result> results, String url){
        this.results = results;
        this.url = url;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        client = new OkHttpClient();

    }
    @Override
    protected Boolean doInBackground(QuantizedTensor... qxs) {
        QuantizedTensor qx = qxs[0];
        Headers requestHeader = new Headers.Builder()
                .add("w", Integer.toString(qx.originalWidth))
                .add("h", Integer.toString(qx.originalHeight))
                .add("image_id", "000000133244.jpg")
                .add("zero_point", Integer.toString(qx.zero_point))
                .add("scale", Float.toString(qx.scale))
                .build();
        RequestBody requestBody = RequestBody.create(qx.qx);

        Request request = new Request.Builder()
                .url(url)
                .headers(requestHeader)
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()){
                return false;
            }
            parseResults(response);
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        return true;
    }

    private void parseResults(Response response) throws IOException {
//        System.out.println(response.body().string());

        final String responseBody = response.body().string();
        JsonObject jsonObject =  JsonParser.parseString(responseBody).getAsJsonObject();
        JsonArray data = jsonObject.getAsJsonArray("data");
        for(JsonElement obj: data){
            Result r = parseJSONResult(obj.getAsJsonObject());
            System.out.println(r);
            results.add(r);
        }
    }
    private Result parseJSONResult(JsonObject obj){
        int[] numbers = jsonParser.fromJson(obj.getAsJsonObject().get("bbox"), int[].class);

        return new Result(
                obj.get("class").getAsInt(),
                obj.get("score").getAsFloat(),
                new Rect(numbers[0], numbers[1], numbers[2], numbers[3]));
    }
}
