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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_map);
        mImageText = findViewById(R.id.textCurrentImage);
        progressBar = findViewById(R.id.progressBarMap);
        chronometer = findViewById(R.id.simpleChronometer);

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

        ApiHandler apiHandler = new ApiHandler();
        apiHandler.clearServerMAP();
        chronometer.setBase(SystemClock.elapsedRealtime());
        chronometer.start();
        for (int i = 0; i < images.length; i++) {
            mImageText.setText(images[i]);
            progressBar.setProgress(i * (progressBar.getMax() - progressBar.getMin()) / images.length);
            try {
                Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("coco_images/" + images[i]));
                QuantizedTensor qx = moduleWrapper.run(bitmap, images[i]);
                apiHandler.postSplitTensor(qx);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        chronometer.stop();
        apiHandler.getServerMAP();

    }
}
