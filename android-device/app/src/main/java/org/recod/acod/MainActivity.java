// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.recod.acod;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;


public class MainActivity extends AppCompatActivity implements Runnable {
    private int mImageIndex = 0;
    private String imageId = "000000177893.jpg";

    private ImageView mImageView;
    private ResultView mResultView;
    private Button mButtonDetect;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
//    private Module encoder = null;
//    private Module decoder = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;


    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);


        Dataset.LoadFromDisk(getApplicationContext());
        Dataset dataset = Dataset.getInstance();
        while(!dataset.isReady()){
            System.out.println("Waiting obb mount. Are you sure the file is there?");
        }; // Wait obb load

        try {
            File imageFile = dataset.getRandomImage();
            imageId = imageFile.getName();
            System.out.println(imageFile.canRead());
//            String path = imageFile.getAbsolutePath();
            FileInputStream fileInputStream = new FileInputStream(imageFile);
            mBitmap = BitmapFactory.decodeStream(fileInputStream);
        } catch (Exception e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);


        final Button buttonLive = findViewById(R.id.liveButton);
        buttonLive.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
              final Intent intent = new Intent(MainActivity.this, ObjectDetectionActivity.class);
              startActivity(intent);
            }
        });

        final Button buttonMap = findViewById(R.id.mapButton);
        buttonMap.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                final Intent intent = new Intent(MainActivity.this, MaPActivity.class);
                startActivity(intent);
            }
        });



        mButtonDetect = findViewById(R.id.detectButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonDetect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonDetect.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonDetect.setText(getString(R.string.run_model));

                mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.mInputWidth;
                mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.mInputHeight;

                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)mImageView.getWidth() / mBitmap.getWidth() : (float)mImageView.getHeight() / mBitmap.getHeight());
                mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)mImageView.getHeight() / mBitmap.getHeight() : (float)mImageView.getWidth() / mBitmap.getWidth());

                mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
                mStartY = (mImageView.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        try {
            String modulePath = MainActivity.assetFilePath(getApplicationContext(), "effd2_encoder.ptl");
            mModule = LiteModuleLoader.load(modulePath);
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes_efficientDet.txt")));

            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            PrePostProcessor.mClasses = new String[classes.size()];
            classes.toArray(PrePostProcessor.mClasses);
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
                        matrix.postRotate(90.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
                                matrix.postRotate(90.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
    }


    @Override
    public void run() {
        // TODO: Document differences in preprocessing from PIL to TensorImageUtils
//        Bitmap resizedBitmap = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
        mModule.runMethod("set_width", IValue.from(1.0f));
        IValue outputTuple = mModule.forward(IValue.from(inputTensor));
        final Tensor outputTensor = outputTuple.toTensor();


        mModule.runMethod("set_width", IValue.from(0.5f));
        outputTuple = mModule.forward(IValue.from(inputTensor));
        final Tensor outputTensor2 = outputTuple.toTensor();


        ApiHandler apiHandler = new ApiHandler();

//        QuantizedTensor qbitmap = new QuantizedTensor(resizedBitmap);
//        qbitmap.originalWidth = mBitmap.getWidth();
//        qbitmap.originalHeight = mBitmap.getHeight();
//        try {
//            apiHandler.postImageTensor(qbitmap);
//        } catch (ExecutionException e) {
//            e.printStackTrace();
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }

        QuantizedTensor qx = new QuantizedTensor(outputTensor, 8, imageId);
        qx.originalWidth = mBitmap.getWidth();
        qx.originalHeight = mBitmap.getHeight();
        try {
            apiHandler.postSplitTensor(qx);
        } catch (ExecutionException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

//        final float[] outputs = outputTensor.getDataAsFloatArray();
        ResultScaler resultScaler = new ResultScaler(
                mImgScaleX, mImgScaleY, mIvScaleX,
                mIvScaleY, mStartX, mStartY);

        final ArrayList<Result> results =  resultScaler.RescaleResults(apiHandler.lastResults);
        runOnUiThread(() -> {
            mButtonDetect.setEnabled(true);
            mButtonDetect.setText(getString(R.string.detect));
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            mResultView.setResults(results);
            mResultView.invalidate();
            mResultView.setVisibility(View.VISIBLE);
        });
    }

}
