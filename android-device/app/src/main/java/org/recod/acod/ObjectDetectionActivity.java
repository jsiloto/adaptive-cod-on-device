package org.recod.acod;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;
import android.view.TextureView;
import android.view.ViewStub;

import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;

public class ObjectDetectionActivity extends AbstractCameraXActivity<ObjectDetectionActivity.AnalysisResult> {
    private ResultView mResultView;
    private PytorchModuleWrapper moduleWrapper = null;
    private ApiHandler apiHandler = new ApiHandler();


    static class AnalysisResult {
        private final ArrayList<Result> mResults;

        public AnalysisResult(ArrayList<Result> results) {
            mResults = results;
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_object_detection;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.resultView);
        return ((ViewStub) findViewById(R.id.object_detection_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);
    }

    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    @Override
    @Nullable
    protected void analyzeImageAndUpdateUI(ImageProxy image, int rotationDegrees) {
        try {
            if (moduleWrapper == null) {
                String modulePath = MainActivity.assetFilePath(getApplicationContext(), "effd2_encoder.ptl");
                moduleWrapper = new PytorchModuleWrapper(modulePath);
                moduleWrapper.setWidth(0.25f);
            }
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            return;
        }
        Bitmap bitmap = imgToBitmap(image.getImage());
        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);

        QuantizedTensor qx = moduleWrapper.run(bitmap, "");

        float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;
        mImgScaleX = (float)bitmap.getWidth() / PrePostProcessor.mInputWidth;
        mImgScaleY = (float)bitmap.getHeight() / PrePostProcessor.mInputHeight;

        mIvScaleX = (bitmap.getWidth() > bitmap.getHeight() ? (float)mResultView.getWidth() / bitmap.getWidth() : (float)mResultView.getHeight() / bitmap.getHeight());
        mIvScaleY  = (bitmap.getHeight() > bitmap.getWidth() ? (float)mResultView.getHeight() / bitmap.getHeight() : (float)mResultView.getWidth() / bitmap.getWidth());

        mStartX = (mResultView.getWidth() - mIvScaleX * bitmap.getWidth())/2;
        mStartY = (mResultView.getHeight() -  mIvScaleY * bitmap.getHeight())/2;

        class drawResults implements AsyncPostTensor.onPostExecuteCallback {
            @Override
            public void execute(ArrayList<Result> results) {
                runOnUiThread(() -> {
                    ResultScaler resultScaler = new ResultScaler(
                            mImgScaleX, mImgScaleY, mIvScaleX,
                            mIvScaleY, mStartX, mStartY);
                    final ArrayList<Result> scaledResults =
                            resultScaler.RescaleResults(results);
                    mResultView.setResults(scaledResults);
                    mResultView.invalidate();
                });
            }
        }
        apiHandler.postSplitTensor(qx, new drawResults());
    }
}
