package org.pytorch.demo.objectdetection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.widget.Chronometer;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.Module;

import java.io.IOException;

public class MaPActivity extends AppCompatActivity implements Runnable {
    private TextView mImageText;
    private ProgressBar progressBar;
    private Thread thread;
    private String[] images;
    private Module mModule = null;
    private String modulePath;
    private PytorchModuleWrapper moduleWrapper;
    private Chronometer chronometer;
    private ApiHandler apiHandler = new ApiHandler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_map);
        mImageText = findViewById(R.id.textCurrentImage);
        progressBar = findViewById(R.id.progressBarMap100);
        chronometer = findViewById(R.id.chronometer100);

        try {
            modulePath = MainActivity.assetFilePath(getApplicationContext(), "effd2_encoder.ptl");
            images = getAssets().list("coco_images/");
        } catch (IOException e) {
            e.printStackTrace();
        }
        moduleWrapper = new PytorchModuleWrapper(modulePath);

        thread = new Thread(MaPActivity.this);
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

    }

    private void run_at_alpha(float alpha, ProgressBar progressBar, Chronometer chronometer) {
        moduleWrapper.setWidth(alpha);
        apiHandler.clearServerMAP();
        chronometer.setBase(SystemClock.elapsedRealtime());
        chronometer.start();
        int max_images = images.length;
//        max_images = 20;

        for (int i = 0; i < max_images; i++) {
            mImageText.setText(images[i]);
            try {
                Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("coco_images/" + images[i]));
                QuantizedTensor qx = moduleWrapper.run(bitmap, images[i]);
                apiHandler.postSplitTensor(qx);
            } catch (Exception e) {
                e.printStackTrace();
            }
            progressBar.setProgress((i + 1) * (progressBar.getMax() - progressBar.getMin()) / max_images);
        }

        try {
            while(apiHandler.response_counter[0] < max_images){
                System.out.println("Waiting all requests to resolve");
            }
            chronometer.stop();
            String results = apiHandler.getServerMAP();
            apiHandler.postData(results, String.format("device_%3d", (int) (alpha * 100)));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
