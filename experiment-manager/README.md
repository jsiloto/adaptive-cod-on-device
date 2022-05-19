adb shell am force-stop com.jsiloto.myapplication 
adb logcat -c
adb shell am start -n com.jsiloto.myapplication/.DisplayMessageActivity -e TEST "abcds"
adb logcat -d -e "ExperimentOutput"



# Set Up the Board

# Generate OBB
- Donwload from
- https://drive.google.com/file/d/1UkIqdwCDif_jv22q1w2xpago44yAcOH6/view?usp=sharing

# How to Install the APK

# Run The Script