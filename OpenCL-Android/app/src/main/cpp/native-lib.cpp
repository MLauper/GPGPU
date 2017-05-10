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

    // Get Platform Class and Constructor
    jclass cls = env->FindClass("ch/collab/android/opencldemo/Platform");
    jmethodID constructor = env->GetMethodID(cls, "<init>", "()V");

    // Get field IDs
    jfieldID platformProfileField = env->GetFieldID(cls, "platformProfile", "Ljava/lang/String;");
    jfieldID platformVersionField = env->GetFieldID(cls, "platformVersion", "Ljava/lang/String;");
    jfieldID platformNameField = env-> GetFieldID(cls, "platformName", "Ljava/lang/String;");
    jfieldID platformVendorField = env->GetFieldID(cls, "platformVendor", "Ljava/lang/String;");
    jfieldID platformExtensionsField = env->GetFieldID(cls, "platformExtensions", "Ljava/lang/String;");

    // Create Array of platform objects in java and fill in all fields
    auto ret = (jobjectArray)env->NewObjectArray(numOfPlatforms, cls, env->NewObject(cls, constructor));
    for (int i = 0; i < numOfPlatforms; i++)
    {
        jobject obj = env->GetObjectArrayElement(ret, i);

        const char* platformProfile = convertToString(platforms[i]->getClPlatformProfile());
        env->SetObjectField(obj, platformProfileField, env->NewStringUTF(platformProfile));

        const char* platformVersion = convertToString(platforms[i]->getClPlatformVersion());
        env->SetObjectField(obj, platformVersionField, env->NewStringUTF(platformVersion));

        const char* platformName = convertToString(platforms[i]->getClPlatformName());
        env->SetObjectField(obj, platformNameField, env->NewStringUTF(platformName));

        const char* platformVendor = convertToString(platforms[i]->getClPlatformVendor());
        env->SetObjectField(obj, platformVendorField, env->NewStringUTF(platformVendor));

        const char* platformExtensions = convertToString(platforms[i]->getClPlatformExtensions());
        env->SetObjectField(obj, platformExtensionsField, env->NewStringUTF(platformExtensions));
    }

    return ret;
}