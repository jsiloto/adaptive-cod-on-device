package org.recod.acod;

import android.content.Intent;
import android.os.Bundle;
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


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Get Experiment Config from Extras
        // Extras should be passed via adb shell am start [intent] -e [extra]
        String model = "";
        float alpha = 1.0f;
        boolean wifiOn = false;
        boolean useDummyModel = false;
        Intent intent = getIntent();
        Bundle extras = intent.getExtras();
        if (extras != null) {
            model = extras.getString("model", model);
            alpha = extras.getFloat("alpha", alpha);
            wifiOn = extras.getBoolean("wifi", wifiOn);
        }
        if(model.isEmpty()){
            useDummyModel=true;
        }

        // Acquire User Interface
        setContentView(R.layout.activity_experiment);
        textConfigs = findViewById(R.id.textConfigs);
        textResults = findViewById(R.id.textResults);
        progressBar = findViewById(R.id.progressBarExperiment);
        chronometer = findViewById(R.id.chronometer);

        // Load Experiment
        textConfigs.setText(String.format("%s: alpha=%f, wifi=%s",
                model, alpha, wifiOn ? "on" : "off"));
        try {
            String modulePath = Helper.assetFilePath(this.getApplicationContext(), model);
            moduleWrapper = new PytorchModuleWrapper(modulePath, useDummyModel);
            moduleWrapper.setWidth(alpha);
        } catch (IOException e) {
            e.printStackTrace();
        }


        // Start
        thread = new Thread(this);
        thread.start();
    }

    @Override
    public void onBackPressed() {
        thread.interrupt();
        this.finish();
        super.onBackPressed();
    }


    @Override
    public void run() {
        String results = "";
        try {
            results += "\n" + new LatencyExperiment("effd2_encoder.ptl", this.getApplicationContext()).run();
            textResults.setText(results);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
