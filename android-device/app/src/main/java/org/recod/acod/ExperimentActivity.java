package org.recod.acod;

import android.os.Bundle;
import android.os.PowerManager;
import android.widget.Chronometer;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.Module;

import java.io.IOException;

public class ExperimentActivity extends AppCompatActivity implements Runnable {
    private TextView textConfigs, textResults;
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
        setContentView(R.layout.activity_experiment);
        textConfigs = findViewById(R.id.textConfigs);
        textResults = findViewById(R.id.textResults);
        progressBar = findViewById(R.id.progressBarExperiment);
        chronometer = findViewById(R.id.chronometer);
        thread = new Thread(this);
        thread.start();
    }

    @Override
    public void onBackPressed() {
        // Below, everything I am just
        thread.interrupt();
        this.finish();
        Thread.currentThread().interrupt();
        super.onBackPressed();
    }


    @Override
    public void run() {
        String results = "";
        try {
//            results += "\n" + new LatencyExperiment("efficientdet_wightman.ptl", this.getApplicationContext()).run();
//            textResults.setText(results);
            //TODO(jsiloto): Add width and heigth
 //           results += "\n" + new LatencyExperiment("yolov5s.torchscript.ptl", this.getApplicationContext()).run();
  //          textResults.setText(results);
   //         results += "\n" + new LatencyExperiment("effd2_full.ptl", this.getApplicationContext()).run();
    //        textResults.setText(results);
            results += "\n" + new LatencyExperiment("effd2_encoder.ptl", this.getApplicationContext()).run();
            textResults.setText(results);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
