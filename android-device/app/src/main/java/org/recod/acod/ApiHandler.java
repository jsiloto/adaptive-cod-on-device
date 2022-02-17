package org.recod.acod;


public class ApiHandler {
    private String url = "http://192.168.15.172:5000/";
    private boolean dummy = false;

    ApiHandler() {
    }

    ApiHandler(String url) {
        this.url = url;
    }

    ApiHandler(String url, boolean dummy) {
        this.url = url;
        this.dummy = dummy;
    }


    public void postSplitTensor(QuantizedTensor qx, AsyncPostTensor.onPostExecuteCallback callback) {
        if(!dummy){
            new AsyncPostTensor(url + "split", callback).execute(qx);
        }
    }

    public void clearServerMAP() {
        if(!dummy){
            MapAPI.clearServerState(url + "map");
        }
    }

    public String getServerMAP() {
        String result = "";
        if(!dummy){
            result = MapAPI.getServerState(url + "map");
        }
        return result;
    }

//    public Boolean postData(String data, String endpoint) throws IOException {
//        OkHttpClient client = new OkHttpClient();
//        RequestBody requestBody = RequestBody.create(data.getBytes());
//        Request request = new Request.Builder()
//                .url(url + "data/" + endpoint)
//                .post(requestBody)
//                .build();
//
//        Response response = client.newCall(request).execute();
//        if (!response.isSuccessful()) {
//            return false;
//        } else {
//            return true;
//        }
//    }

}
