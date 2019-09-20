cc_library(
    name = "conv2d",
    srcs = ["conv2d.cpp", 
            "opencl.cpp",
            "options.cpp",
           ],                                               
    hdrs = ["inc/AOCLUtils/aocl_utils.h",
            "inc/AOCLUtils/opencl.h",
            "conv2d.h",
            "inc/AOCLUtils/scoped_ptrs.h",
            "inc/AOCLUtils/options.h",
            "include/CL/opencl.h",
            "include/CL/cl_d3d10.h", 
            "include/CL/cl_d3d11.h",
            "include/CL/cl_dx9_media_sharing.h",
            "include/CL/cl_dx9_media_sharing_intel.h",
            "include/CL/cl_egl.h",
            "include/CL/cl_ext.h",
            "include/CL/cl_ext_intelfpga.h",
            "include/CL/cl_ext_intel.h",
            "include/CL/cl_gl_ext.h",
            "include/CL/cl_gl.h",
            "include/CL/cl.h",
            "include/CL/cl_platform.h",
            "include/CL/cl_va_api_media_sharing_intel.h",
            "include/CL/cl_version.h",
           ],                                                                           
    copts = ["-I./include", "-L./lib", "-O2", " -fstack-protector", " -D_FORTIFY_SOURCE=2", "-Wformat -Wformat-security", "-fPIE", "-fPIC"],
    linkopts = ["-lstdc++", "-L./lib", "-L./lib/ -lacl_emulator_kernel_rt", "-lalteracl", "-lalterahalmmd", "-laoc_cosim_mmd", "-laoc_cosim_msim", "-lelf", "-lOpenCL", "-lz"],                                      
)