# TensorLight
TensorLight is a light-weight tensor operation library for C and CUDA.

## Prerequisites
Required packages can be installed using the following command:

```
sudo apt-get install build-essential perl git
```

If you want to build with CUDA support, you also have to install CUDA 8.0
or later according to their website [CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive).
Remember to put `nvcc` (usually in /usr/local/cuda/bin) in environment variable `PATH`.

## Building and Installation
1.  Clone this repository to your local directory.

    ```
    cd <my_working_directory>
    git clone https://github.com/zhaozhixu/TensorLight.git
    cd TensorLight
    ```

2.  Build and install

    Use `make` to compile the library and run the tests. Then `make install`
    to install the library files and headers into the installation directory,
    or `sudo make install` if you don't have the permissions with the
    installation directory.
    
    There are some options to custom your building and installation process.
    You can just append those options after `make`, such as
    
    ```
    make WITH_CUDA=1
    sudo make install INSTALL_DIR=/usr/local
    ```
 
    `make` options:
    * `WITH_CUDA=<option>` set to 1 if you want to build with CUDA
    * `CUDA_INSTALL_DIR=<cuda install directory>` default is /usr/local/cuda
    
    `make install` and `make uninstall` options:
    * `INSTALL_DIR=<installation directory>` default is /usr/local
    * `PKGCONFIG_DIR=<pkgconfig directory>` default is /usr/lib/pkgconfig

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
