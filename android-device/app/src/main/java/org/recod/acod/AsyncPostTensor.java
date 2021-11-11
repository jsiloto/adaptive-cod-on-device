package org.recod.acod;

import android.os.AsyncTask;

import com.google.gson.Gson;

import java.util.ArrayList;

import okhttp3.OkHttpClient;
import okhttp3.Response;

public class AsyncPostTensor extends AsyncTask<QuantizedTensor, Void, ArrayList<Result>> {
    private OkHttpClient client;
    private Response response;
    private String url;
    private Gson jsonParser = new Gson();
    private FrameTracker frameTracker;
    private onPostExecuteCallback callback;
    private PostTensor postTensor;

    public AsyncPostTensor(String url, onPostExecuteCallback callback) {
        postTensor = new PostTensor(url);
        this.callback = callback;
    }

    public interface onPostExecuteCallback {
        void execute(ArrayList<Result> results);
    }

    @Override
    protected ArrayList<Result> doInBackground(QuantizedTensor... qxs) {
        return postTensor.post(qxs);
    }

    @Override
    protected void onPostExecute(ArrayList<Result> results) {
        callback.execute(results);
    }


}
