package org.recod.acod;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.PowerManager;
import android.os.SystemClock;
import android.widget.Chronometer;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.Module;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;


public class MaPActivity extends AppCompatActivity implements Runnable {
    private TextView mImageText, mRoundTripText;
    private ProgressBar progressBar;
    private Thread thread;
    private String[] images;
    private Module mModule = null;
    private String modulePath;
    private PytorchModuleWrapper moduleWrapper;
    private Chronometer chronometer;
    private ApiHandler apiHandler = new ApiHandler();
    private long dnnTime;
    private PowerManager.WakeLock wakeLock;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Dataset.LoadFromDisk(getApplicationContext());
        Dataset dataset = Dataset.getInstance();
        while (!dataset.isReady()) {
            System.out.println("Waiting obb mount. Are you sure the file is there?");
        }
        ; // Wait obb load


        setContentView(R.layout.activity_map);
        mImageText = findViewById(R.id.textCurrentImage);
        mRoundTripText = findViewById(R.id.textRoundTrip);
        progressBar = findViewById(R.id.progressBarMap100);
        chronometer = findViewById(R.id.chronometer100);

        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MyApp::MyWakelockTag");

        try {
            modulePath = MainActivity.assetFilePath(getApplicationContext(), "effd2_encoder.ptl");
            images = getAssets().list("coco_images/");
        } catch (IOException e) {
            e.printStackTrace();
        }
        moduleWrapper = new PytorchModuleWrapper(modulePath);

        thread = new Thread(MaPActivity.this);
        thread.setPriority(Thread.MAX_PRIORITY);
        thread.start();
    }

    @Override
    public void onBackPressed() {
        // Below, everything I am just
        thread.interrupt();
        MaPActivity.this.finish();
        Thread.currentThread().interrupt();
        super.onBackPressed();
    }


    @Override
    public void run() {
        wakeLock.acquire();

        progressBar = findViewById(R.id.progressBarMap025);
        chronometer = findViewById(R.id.chronometer025);
        run_at_alpha(0.25f, progressBar, chronometer);

        progressBar = findViewById(R.id.progressBarMap050);
        chronometer = findViewById(R.id.chronometer050);
        run_at_alpha(0.50f, progressBar, chronometer);

        progressBar = findViewById(R.id.progressBarMap075);
        chronometer = findViewById(R.id.chronometer075);
        run_at_alpha(0.75f, progressBar, chronometer);

        progressBar = findViewById(R.id.progressBarMap100);
        chronometer = findViewById(R.id.chronometer100);
        run_at_alpha(1.00f, progressBar, chronometer);

        wakeLock.release();
    }

    private void run_at_alpha(float alpha, ProgressBar progressBar, Chronometer chronometer) {
//        apiHandler.clearServerMAP();
        moduleWrapper.setWidth(alpha);

        File[] imageList = Dataset.getInstance().getFileList();
        int max_images = imageList.length;
        max_images = 25;

        /******************** Warmup ****************/
        try {
            String imageId = imageList[0].getName();
            FileInputStream stream = new FileInputStream(imageList[0]);
            Bitmap bitmap = BitmapFactory.decodeStream(stream);
            QuantizedTensor qx = moduleWrapper.run(bitmap, imageId);
        } catch (IOException e) {
            System.out.println("Error processing tensor");
            e.printStackTrace();
        }


        FrameTracker frameTracker = new FrameTracker();
        class TrackerCallback implements AsyncPostTensor.onPostExecuteCallback {
            @Override
            public void execute(ArrayList<Result> results) {
                frameTracker.RegisterFrameEnd();
            }
        }

        chronometer.setBase(SystemClock.elapsedRealtime());
        chronometer.start();
        for (int i = 0; i < max_images; i++) {
            String imageId = imageList[i].getName();
            frameTracker.RegisterNewFrame();
            mImageText.setText(String.format("%s [%d/%d]", imageId, i, max_images));
            mRoundTripText.setText(frameTracker.statistics());
            try {
                FileInputStream stream = new FileInputStream(imageList[i]);
                Bitmap bitmap = BitmapFactory.decodeStream(stream);

                frameTracker.RegisterDnnStart();
                QuantizedTensor qx = moduleWrapper.run(bitmap, imageId);

                frameTracker.RegisterRequest();
//                apiHandler.postSplitTensor(qx, new TrackerCallback());
                stream.close();

                progressBar.setProgress((i + 1) * (progressBar.getMax() - progressBar.getMin()) / max_images);
            } catch (IOException e) {
                System.out.println("Error processing tensor");
                e.printStackTrace();
            }
        }

        chronometer.stop();
//        try {
//            while (!frameTracker.isDone()) {
//                System.out.println("Waiting all requests to resolve");
//            }
//            chronometer.stop();
//            String results = apiHandler.getServerMAP();
//            JsonObject jsonObject = JsonParser.parseString(results).getAsJsonObject();
//            jsonObject.add("performance", JsonParser.parseString(String.valueOf(frameTracker)));
////            apiHandler.postData(jsonObject.toString(), String.format("device_%3d.json", (int) (alpha * 100)));
//        } catch (Exception e) {
//            System.out.println("Error processing results");
//            e.printStackTrace();
//        }
    }
}
