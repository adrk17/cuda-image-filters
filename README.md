# CUDA Image Filters

## Installation guide

### Step 1: Install vcpkg

#### Clone the vcpkg Repository

Open **Command Prompt** and run the following command:

```sh
git clone https://github.com/microsoft/vcpkg.git
```

#### Bootstrap vcpkg

Navigate to the cloned vcpkg directory and run the bootstrap script:

```sh
cd .\vcpkg
bootstrap-vcpkg.bat
```

This will build vcpkg and set up the necessary environment.

### Step 2: Integrate vcpkg with Visual Studio

Run the following command from the vcpkg installation foder to integrate vcpkg with Visual Studio:

```sh
./vcpkg.exe integrate install
```

This allows Visual Studio to automatically detect and use libraries installed via vcpkg.

### Step 3: Build the project

vcpkg should automatically download and install the required dependencies while building the project in Visual Studio, based on the `vcpkg.json` manifest file.

### Step 4: Not visible opencv2 workaround

If the paths to the OpenCV libraries are not visible in Visual Studio, you should go to the `vcpkg_installed/x64-windows/x64-windows/include/opencv4` folder and move the `opencv2` folder outside `opencv4`, to `vcpkg_installed/x64-windows/x64-windows/include` folder. This is a workaround for the issue where OpenCV libraries are not detected properly in Visual Studio because the `opencv2` folder is nested inside `opencv4`. This should make the OpenCV libraries visible in Visual Studio. You can also remove the `opencv4` folder if you want to keep the structure clean.

Side note: The best practice is to keep the `opencv2` folder inside `opencv4` and use the correct include paths in your project settings, but this workaround is a quick fix for the issue and i couldn't find a better working solution.
