package ch.collab.android.opencldemo;

public class Device {
    private String ID;
    private String available;
    private String name;
    private String platform;
    private String profile;
    private String version;
    private String type;
    private String vendorID;
    private String driverVersion;
    private String globalMemCacheSize;
    private String globalMemCacheType;
    private String globalMemSize;
    private String localMemSize;
    private String localMemType;
    private String deviceExtensions;

    public String getID() {
        return ID;
    }

    public String getAvailable() {
        return available;
    }

    public String getName() {
        return name;
    }

    public String getPlatform() {
        return platform;
    }

    public String getProfile() {
        return profile;
    }

    public String getVersion() {
        return version;
    }

    public String getType() {
        return type;
    }

    public String getVendorID() {
        return vendorID;
    }

    public String getDriverVersion() {
        return driverVersion;
    }

    public String getGlobalMemCacheSize() {
        return globalMemCacheSize;
    }

    public String getGlobalMemCacheType() {
        return globalMemCacheType;
    }

    public String getGlobalMemSize() {
        return globalMemSize;
    }

    public String getLocalMemSize() {
        return localMemSize;
    }

    public String getLocalMemType() {
        return localMemType;
    }

    public String getDeviceExtensions() {
        return deviceExtensions;
    }
}
