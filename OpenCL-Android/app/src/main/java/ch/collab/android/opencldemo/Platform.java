package ch.collab.android.opencldemo;

public class Platform {
    private String platformProfile;
    private String platformVersion;
    private String platformName;
    private String platformVendor;
    private String platformExtensions;


    public Platform(){

    }

    public String getPlatformVendor() {
        return platformVendor;
    }

    public String getPlatformProfile() {
        return platformProfile;
    }

    public String getPlatformVersion() {
        return platformVersion;
    }

    public String getPlatformName() {
        return platformName;
    }

    public String getPlatformExtensions() {
        return platformExtensions;
    }
    /*
    cl_platform_id getId();
    char* getClPlatformProfile();
    char* getClPlatformVersion();
    char* getClPlatformName();
    char* getClPlatformVendor();
    char* getClPlatformExtensions();
    std::vector<opencl_helpers::Device*> getDevices();
    */
}
