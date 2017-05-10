#include <jni.h>
#include <string>
#include <sstream>
#include <opencl_helpers.h>
#include <chrono>
#include <thread>

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
    jfieldID deviceIDField = env->GetFieldID(clsDevice, "ID", "Ljava/lang/String;");
    jfieldID deviceAvailableField = env->GetFieldID(clsDevice, "available", "Ljava/lang/String;");
    jfieldID deviceNameField = env->GetFieldID(clsDevice, "name", "Ljava/lang/String;");
    jfieldID devicePlatformField = env->GetFieldID(clsDevice, "platform", "Ljava/lang/String;");
    jfieldID deviceProfileField = env->GetFieldID(clsDevice, "profile", "Ljava/lang/String;");
    jfieldID deviceVersionField = env->GetFieldID(clsDevice, "version", "Ljava/lang/String;");
    jfieldID deviceTypeField = env->GetFieldID(clsDevice, "type", "Ljava/lang/String;");
    jfieldID deviceVendorIDField = env->GetFieldID(clsDevice, "vendorID", "Ljava/lang/String;");
    jfieldID deviceDriverVersionField = env->GetFieldID(clsDevice, "driverVersion", "Ljava/lang/String;");
    jfieldID deviceGlobalMemCacheSizeField = env->GetFieldID(clsDevice, "globalMemCacheSize", "Ljava/lang/String;");
    jfieldID deviceGlobalMemCacheTypeField = env->GetFieldID(clsDevice, "globalMemCacheType", "Ljava/lang/String;");
    jfieldID deviceGlobalMemSizeField = env->GetFieldID(clsDevice, "globalMemSize", "Ljava/lang/String;");
    jfieldID deviceLocalMemSizeField = env->GetFieldID(clsDevice, "localMemSize", "Ljava/lang/String;");
    jfieldID deviceLocalMemTypeField = env->GetFieldID(clsDevice, "localMemType", "Ljava/lang/String;");
    jfieldID deviceDeviceExtensionsField = env->GetFieldID(clsDevice, "deviceExtensions", "Ljava/lang/String;");



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
            jobject objDevice = env->GetObjectArrayElement(jDevices, j);

            auto deviceId = convertToString(devices[j]->getId());
            env->SetObjectField(objDevice, deviceIDField, env->NewStringUTF(deviceId));

            auto device = cl::Device(devices[j]->getId());
            auto deviceAvailable = convertToString(device.getInfo<CL_DEVICE_AVAILABLE>());
            env->SetObjectField(objDevice, deviceAvailableField, env->NewStringUTF(deviceAvailable));
            auto deviceName = convertToString(device.getInfo<CL_DEVICE_NAME>());
            env->SetObjectField(objDevice, deviceNameField, env->NewStringUTF(deviceName));
            auto devicePlatform = convertToString(device.getInfo<CL_DEVICE_PLATFORM>());
            env->SetObjectField(objDevice, devicePlatformField, env->NewStringUTF(devicePlatform));
            auto deviceProfile = convertToString(device.getInfo<CL_DEVICE_PROFILE>());
            env->SetObjectField(objDevice, deviceProfileField, env->NewStringUTF(deviceProfile));
            auto deviceVersion = convertToString(device.getInfo<CL_DEVICE_VERSION>());
            env->SetObjectField(objDevice, deviceVersionField, env->NewStringUTF(deviceVersion));
            auto deviceType = convertToString(device.getInfo<CL_DEVICE_TYPE>());
            env->SetObjectField(objDevice, deviceTypeField, env->NewStringUTF(deviceType));
            auto deviceVendorID = convertToString(device.getInfo<CL_DEVICE_VENDOR_ID>());
            env->SetObjectField(objDevice, deviceVendorIDField, env->NewStringUTF(deviceVendorID));
            auto deviceDriverVersion = convertToString(device.getInfo<CL_DRIVER_VERSION>());
            env->SetObjectField(objDevice, deviceDriverVersionField, env->NewStringUTF(deviceDriverVersion));
            auto deviceGlobalMemCacheSize = convertToString(device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>());
            env->SetObjectField(objDevice, deviceGlobalMemCacheSizeField, env->NewStringUTF(deviceGlobalMemCacheSize));
            auto deviceGlobalMemCacheType = convertToString(device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>());
            env->SetObjectField(objDevice, deviceGlobalMemCacheTypeField, env->NewStringUTF(deviceGlobalMemCacheType));
            auto deviceGlobalMemSize = convertToString(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
            env->SetObjectField(objDevice, deviceGlobalMemSizeField, env->NewStringUTF(deviceGlobalMemSize));
            auto deviceLocalMemSize = convertToString(device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>());
            env->SetObjectField(objDevice, deviceLocalMemSizeField, env->NewStringUTF(deviceLocalMemSize));
            auto deviceLocalMemType = convertToString(device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>());
            env->SetObjectField(objDevice, deviceLocalMemTypeField, env->NewStringUTF(deviceLocalMemType));
            auto deviceDeviceExtensions = convertToString(device.getInfo<CL_DEVICE_EXTENSIONS>());
            env->SetObjectField(objDevice, deviceDeviceExtensionsField, env->NewStringUTF(deviceDeviceExtensions));

        }

        env->SetObjectField(objPlatform, platformDevicesField, jDevices);
    }

    auto defaultDevice = cl::Device::getDefault();

    return jPlatforms;
}