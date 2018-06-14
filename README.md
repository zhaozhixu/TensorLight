# TensorLight
TensorLight is a light-weight tensor operation library for C and CUDA.

## Prerequisites
The following steps have been tested for Ubuntu 16.04 but should work with
other distros as well. 
Required packages can be installed using the following command:

```
sudo apt-get install build-essential perl git pkg-config check
```

If you want to build with CUDA support, you also have to install CUDA 8.0
or later according to their website [CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive).
Remember to put `nvcc` (usually in `/usr/local/cuda/bin`) in environment variable `PATH`.

## Building and Installation
1.  Clone this repository to your local directory.

    ```
    cd <my_working_directory>
    git clone https://github.com/zhaozhixu/TensorLight.git
    cd TensorLight
    ```

2.  Build and install

    First, configure your installation using:
    
    ```
    chmod +x configure
    ./configure
    ```
    There are options to custom your building and installation process.
    You can append them after `./configure`, such as:
    
    ```
    ./configure --cuda-enable=1
    ```
    Detailed `./configure` options can be printed using `./configure -h`.

    After that, use `make` to compile the library and run the tests. Then `make install`
    to copy the library files and headers into the installation directory,
    or `sudo make install` if you don't have the permissions with that directory.
    
    ```
    make
    sudo make install
    ```

3.  Other `make` options

    Use `make info` to see other `make` options.
    Especially, you can use `make clean` to clean up the build directory and all
    object files, and `make uninstall` to remove library files and headers from
    the installation directory.

## Usage
Include `tl_tensor.h` in your project to use TensorLight functions.

Documentation is coming soon. But it should be familiar if you have experience
with `numpy` in Python.

You can use the following command to get the compilation and linking flags when
building your project.

```
pkg-config --cflags --libs tensorlight
```
