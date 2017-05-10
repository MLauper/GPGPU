#include <jni.h>
#include <string>
#include <sstream>
#include <opencl_helpers.h>
#include "CL/cl.h"

template<typename T>
const char *convertToString(T input) {
    std::ostringstream ss;
    ss << input;
    return ss.str().c_str();
}

extern "C"
JNIEXPORT jstring JNICALL
Java_ch_collab_android_opencldemo_MainActivity_getNumOfPlatforms(
        JNIEnv *env,
        jobject /* this */) {

    auto numOfPlatforms = opencl_helpers::getNumOfPlatforms();

    return env->NewStringUTF(convertToString(numOfPlatforms));
};

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_ch_collab_android_opencldemo_MainActivity_getPlatformNames(
        JNIEnv *env,
        jobject /* this */) {

    auto numOfPlatforms = opencl_helpers::getNumOfPlatforms();
    auto platforms = opencl_helpers::getPlatforms();

    auto ret = (jobjectArray)env->NewObjectArray(numOfPlatforms, env->FindClass("java/lang/String"),env->NewStringUTF(""));

    for (int i = 0; i < numOfPlatforms; i++)
    {
        env->SetObjectArrayElement(ret, i, env->NewStringUTF(convertToString(platforms[i]->getClPlatformName())));
    }

    return ret;
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_ch_collab_android_opencldemo_MainActivity_getPlatforms(
        JNIEnv *env,
        jobject /* this */) {

    auto numOfPlatforms = opencl_helpers::getNumOfPlatforms();
    auto platforms = opencl_helpers::getPlatforms();

    // Get Java classes and constructors
    jclass clsPlatform = env->FindClass("ch/collab/android/opencldemo/Platform");
    jmethodID constructorPlatform = env->GetMethodID(clsPlatform, "<init>", "()V");
    jclass clsDevice = env->FindClass("ch/collab/android/opencldemo/Device");
    jmethodID constructorDevice = env->GetMethodID(clsPlatform, "<init>", "()V");

    // Get field IDs
    jfieldID platformProfileField = env->GetFieldID(clsPlatform, "platformProfile", "Ljava/lang/String;");
    jfieldID platformVersionField = env->GetFieldID(clsPlatform, "platformVersion", "Ljava/lang/String;");
    jfieldID platformNameField = env-> GetFieldID(clsPlatform, "platformName", "Ljava/lang/String;");
    jfieldID platformVendorField = env->GetFieldID(clsPlatform, "platformVendor", "Ljava/lang/String;");
    jfieldID platformExtensionsField = env->GetFieldID(clsPlatform, "platformExtensions", "Ljava/lang/String;");
    jfieldID platformDevicesField = env->GetFieldID(clsPlatform, "devices", "[ch/collab/android/opencldemo/Device");
    jfieldID deviceIDField = env->GetFieldID(clsDevice, "deviceID", "Ljava/lang/String;");

    // Create Array of platform objects in java and fill in all fields
    auto jPlatforms = (jobjectArray)env->NewObjectArray(numOfPlatforms, clsPlatform, env->NewObject(clsPlatform, constructorPlatform));
    for (int i = 0; i < numOfPlatforms; i++)
    {
        jobject objPlatform = env->GetObjectArrayElement(jPlatforms, i);

        auto platformProfile = convertToString(platforms[i]->getClPlatformProfile());
        env->SetObjectField(objPlatform, platformProfileField, env->NewStringUTF(platformProfile));

        auto platformVersion = convertToString(platforms[i]->getClPlatformVersion());
        env->SetObjectField(objPlatform, platformVersionField, env->NewStringUTF(platformVersion));

        auto platformName = convertToString(platforms[i]->getClPlatformName());
        env->SetObjectField(objPlatform, platformNameField, env->NewStringUTF(platformName));

        auto platformVendor = convertToString(platforms[i]->getClPlatformVendor());
        env->SetObjectField(objPlatform, platformVendorField, env->NewStringUTF(platformVendor));

        auto platformExtensions = convertToString(platforms[i]->getClPlatformExtensions());
        env->SetObjectField(objPlatform, platformExtensionsField, env->NewStringUTF(platformExtensions));

        auto devices = platforms[i]->getDevices();
        auto numOfDevices = static_cast<int>(devices.size());
        auto jDevices = env->NewObjectArray(numOfDevices, clsDevice, env->NewObject(clsDevice, constructorDevice));
        for (int j = 0; j < numOfDevices; j++){
            auto deviceId = convertToString(devices[j]->getId());
            jobject objDevice = env->GetObjectArrayElement(jDevices, j);
            env->SetObjectField(objDevice, deviceIDField, env->NewStringUTF(deviceId));
        }

        env->SetObjectField(objPlatform, platformDevicesField, jDevices);
    }

    return jPlatforms;
}