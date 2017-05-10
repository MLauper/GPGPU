#include <jni.h>
#include <string>
#include <sstream>
#include <opencl_helpers.h>
#include "CL/cl.h"

extern "C"
JNIEXPORT jstring JNICALL
Java_ch_collab_android_opencldemo_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";

    cl_uint num_platforms;
    auto err = ::clGetPlatformIDs(0, NULL, &num_platforms);

    std::ostringstream ss;
    ss << num_platforms;
    hello = ss.str();

    return env->NewStringUTF(hello.c_str());
}

template <typename T>
const char* convertToString(T input){
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

