package org.recod.acod;
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
    private int [] response_counter;

    public AsyncPostTensor(ArrayList<Result> results, String url, int[] response_counter){
        this.results = results;
        this.url = url;
        this.response_counter = response_counter;
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

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()){
                return false;
            }
            parseResults(response);
            response_counter[0]++;
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
        results.clear();
        for(JsonElement obj: data){
            Result r = parseJSONResult(obj.getAsJsonObject());
            results.add(r);
        }
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
