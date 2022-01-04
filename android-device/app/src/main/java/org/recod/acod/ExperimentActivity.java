package org.recod.acod;

import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;

public class ExperimentActivity extends AppCompatActivity implements Runnable {
    private TextView textConfigs;
    private Thread thread;
    private ApiHandler apiHandler;
    private PytorchModuleWrapper moduleWrapper;
    private ExperimentRunner experimentRunner;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Get Experiment Config from Extras
        // Extras should be passed via adb shell am start [intent] -e [extra]
        String serverUrl = "http://192.168.1.102:5000/";
        String model = "";
        float alpha = 1.0f;
        boolean wifiOn = false;
        boolean useDummyModel = false;

        Intent intent = getIntent();
        Bundle extras = intent.getExtras();
        if (extras != null) {
            serverUrl = extras.getString("server", serverUrl);
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
        textConfigs.setText(String.format("%s: alpha=%f, wifi=%s",
                model, alpha, wifiOn ? "on" : "off"));


        // Load Experiment Assets
        try {
            String modulePath = Helper.assetFilePath(this.getApplicationContext(), model);
            moduleWrapper = new PytorchModuleWrapper(modulePath, useDummyModel);
            moduleWrapper.setWidth(alpha);
            apiHandler = new ApiHandler(serverUrl);
        } catch (IOException e) {
            e.printStackTrace();
        }

        experimentRunner = new ExperimentRunner(moduleWrapper, apiHandler);

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
        experimentRunner.run();
    }

}
