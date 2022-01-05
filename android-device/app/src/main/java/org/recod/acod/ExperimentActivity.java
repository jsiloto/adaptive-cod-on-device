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
    private Dataset dataset;
    private ExperimentRunner experimentRunner;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
//        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
//            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
//        }
        // Get Experiment Config from Extras
        // Extras should be passed via adb shell am start [intent] -e [extra]
        String url = "";
        String model = "";
        String modelPath =  "";
        float alpha = 1.0f;
        boolean useDummyModel = false;
        boolean useDummyWifi = false;

        Intent intent = getIntent();
        Bundle extras = intent.getExtras();
        if (extras != null) {
            url = extras.getString("url", url);
            model = extras.getString("model", model);
            alpha = extras.getFloat("alpha", alpha);
        }
        useDummyModel = model.isEmpty();
        useDummyWifi = url.isEmpty();

        // Acquire User Interface
        setContentView(R.layout.activity_experiment);
        textConfigs = findViewById(R.id.textConfigs);
        textConfigs.setText(String.format("%s: alpha=%f, server=%s",model, alpha, url));

        // Load Experiment Assets
        try {
            if(!useDummyModel){
                modelPath = Helper.assetFilePath(this.getApplicationContext(), model);
            }
            moduleWrapper = new PytorchModuleWrapper(modelPath, useDummyModel);
            moduleWrapper.setWidth(alpha);
            apiHandler = new ApiHandler(url, useDummyWifi);
            Dataset.LoadFromDisk(this.getApplicationContext());
            dataset = Dataset.getInstance();
            while(!dataset.isReady()){
                System.out.println("Waiting obb mount. Are you sure the file is there?");
            }; // Wait obb load
            experimentRunner = new ExperimentRunner(moduleWrapper, apiHandler, dataset);
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
        experimentRunner.run();
    }

}
