package org.recod.acod;

import android.content.Context;
import android.os.Environment;
import android.os.storage.OnObbStateChangeListener;
import android.os.storage.StorageManager;
import android.util.Log;

import java.io.File;
import java.util.Random;

// Dataset Singleton
public class Dataset {
    private static Dataset mInstance= null;
    private String obbPath, mountPath, status;
    private StorageManager storageManager;

    protected Dataset(Context context){
        obbPath = Environment.getExternalStorageDirectory() +
                "/Android/obb/org.recod.acod/main.1.org.recod.acod.obb";
        storageManager = (StorageManager) context.getSystemService(context.STORAGE_SERVICE);
        storageManager.mountObb(obbPath, null, mEventListener);
    }
    public static synchronized void LoadFromDisk(Context context){
        if(null == mInstance){
            mInstance = new Dataset(context);
        }
    }

    private OnObbStateChangeListener mEventListener = new OnObbStateChangeListener() {
        @Override
        public void onObbStateChange(String path, int state) {
            Log.d("MyTAG", "path=" + path + "; state=" + state);
            mInstance.status = String.valueOf(state);
            if (state == OnObbStateChangeListener.MOUNTED) {
                mInstance.mountPath = mInstance.storageManager.getMountedObbPath(mInstance.obbPath);
            } else {
                String message = "Obb Never Mounted. State: " + state;
            }
        }
    };

    public static synchronized Dataset getInstance() {
        return mInstance;
    }

    ///////////////////////////// Instance public Methods ///////////////////////////////

    public boolean isReady(){
        return  storageManager.isObbMounted(obbPath);
    }
    public File getRandomImage(){
        File[] listOfFiles = getFileList();
        int rnd = new Random().nextInt(listOfFiles.length);
        return listOfFiles[rnd];
    }

    public File[] getFileList(){
        String f = storageManager.getMountedObbPath(obbPath);
        File folder = new File(f);
        File[] listOfFiles = folder.listFiles();
        return listOfFiles;
    }

}



