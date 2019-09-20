#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>  
#include "include/CL/opencl.h"
#include "inc/AOCLUtils/aocl_utils.h"

#define MAX_SOURCE_SIZE (0x10000)
bool use_fast_emulator = false;

using namespace aocl_utils;

void cleanup()
{
    
}

int Conv2D( )
{

	/*=================================================
	  define parameters
	  =================================================*/
	cl_platform_id	platform_id = NULL;
	cl_uint			ret_num_platforms=0;
	scoped_array<cl_device_id>	devices_id;
	cl_device_id    device;
	cl_uint			ret_num_devices;
	cl_context		context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem			data_in = NULL;
	cl_mem			data_out = NULL;
	cl_mem			filter_in = NULL;
	cl_program		program = NULL;
	cl_kernel		kernel = NULL;
	size_t			kernel_code_size;
	char			*kernel_str;
	float			*result;
	cl_int			ret;
	FILE			*fp;
	cl_uint			work_dim;
	size_t			global_item_size[3];
	size_t			local_item_size[3];
	/*=================================================
	  define parameters for image, filter, kernels
	  =================================================*/

	int const image_width = 8;			//image width
	int const image_height = 8;			//image height
        int const image_depth = 1;

	int const filter_kernel_width = 3;			//filter kernel size
	int const filter_kernel_height = 3;			//filter kernel size
	int const filter_kernel_depth = 3; //filter kernel depth

        int const conv_width_step = 1;   // conv width step
        int const conv_height_step = 1;   // conv height step

        int const image_new_width = (image_width + conv_width_step -1)/conv_width_step;   //image new width after conv and ceil 
        int const image_new_height = (image_height + conv_height_step -1)/conv_height_step;    //image new height after conv and ceil
	int const image_pad_width = (image_new_width - 1)*conv_width_step + filter_kernel_width - image_width; // should add image_pad_width pixel at width
	int const image_pad_height = (image_new_height - 1)*conv_height_step + filter_kernel_height - image_height; //should add image_pad_height oixal at height

        int const pad_top = image_pad_height/2;
        int const pad_down = image_pad_height - pad_top;
        int const pad_left = image_pad_width/2;
        int const pad_right = image_pad_width - pad_left;

	int point_num = (image_width) * (image_height) * image_depth;
	float data_vecs[point_num];
	float filter_coe[filter_kernel_width*filter_kernel_height*filter_kernel_depth] = { -1.0,0,1.0,-2.0,0,2.0,-1.0,0,1.0,
		-2.0,1.0,0,-1.0,1.0,1.0,0,1.0,2.0,
//                1,0,-2,1,2,1,3,-2,2,
//                -1,1,2,0,1,1,2,1,2,
		3.0,0,-2.0,1.0,-1,2.0,-3.0,1.0,1.0}; //sobel filter: horizontal gradient
	int i, j, z;

        printf("\n");
        printf("filter_coe:\n");
        for (i = 0; i < filter_kernel_depth; i++)
        {
            printf("filter_coe depth[%d]:\n", i);
            for(j = 0; j < filter_kernel_height; j++)
            {
                printf("row[%d]:\t", j);
                for(z = 0; z < filter_kernel_width; z++)
                {
                    printf("%f, \t", filter_coe[i*filter_kernel_height*filter_kernel_width + j*filter_kernel_height + z]);
                }
		printf("\n");
            }
	    printf("\n");
        }
	printf("\n");

	for (i = 0; i < point_num; i++)
	{
		data_vecs[i] = (float)(rand() % 5);
	}

	//display input data
	printf("\n");
	printf("Array data_in:\n");
        for(i = 0; i < image_depth; i++)
        {
            printf("data_in depth[%d]:\n", i);
	    for (j = 0; j < (image_height); j++) 
            {
		printf("row[%d]:\t", j);
		for (z = 0; z < (image_width); z++) 
                {
		    printf("%f,\t", data_vecs[i*(image_width)*(image_height) + j*(image_width) + z]);
		}
		printf("\n");
	    }
	    printf("\n");
        }
	printf("\n");
 
        //do padding
        int padded_num = ((image_width + image_pad_width)*(image_height + image_pad_height)*image_depth);
        float data_padded[padded_num];
	for (i = 0; i < padded_num; i++)
	{
		data_padded[i] = 0;
	}
        for(i = 0; i < image_depth; i++)
        {
	    for (j = 0; j < (image_height); j++) 
            {
		for (z = 0; z < (image_width); z++) 
                {
		    data_padded[i*(image_width + image_pad_width)*(image_height + image_pad_height) + (j+pad_top)*(image_width + image_pad_width) + z + pad_left] = data_vecs[i*(image_width)*(image_height) + j*(image_width) + z];
		}
	    }
        }
        
       
        //input data after padding
	printf("\n");
	printf("Array data_in after padding:\n");
        for(i = 0; i < image_depth; i++)
        {
            printf("data_in depth[%d]:\n", i);
	    for (j = 0; j < (image_pad_height + image_height); j++) 
            {
		printf("row[%d]:\t", j);
		for (z = 0; z < (image_width + image_pad_width); z++) 
                {
		    printf("%f,\t", data_padded[i*(image_width + image_pad_width)*(image_height + image_pad_height) + j*(image_width + image_pad_width) + z]);
		}
		printf("\n");
	    }
	    printf("\n");
        }
	printf("\n");
	/*=================================================
	  load kernel, opencl environment setup
	  create input and output buffer
	  set kernel arguments, excute kernels
	  get final results
	  =================================================*/
	kernel_str = (char *)malloc(MAX_SOURCE_SIZE);
	result = (float *)malloc(image_new_width*image_new_height*filter_kernel_depth * sizeof(float));

	if (use_fast_emulator)
	{
		platform_id = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
	}
	else
	{
		platform_id = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
	}
	if (platform_id == NULL)
	{
		printf("Error: Unable to find Intel(R) FPGA OpenCL Platform. \n");
		return false;
	}

	devices_id.reset(getDevices(platform_id, CL_DEVICE_TYPE_ALL, &ret_num_devices));
	printf("Platform: %s\n", getPlatformName(platform_id).c_str());
	printf("Using %d device(s)\n", ret_num_devices);
	for(unsigned i = 0; i < ret_num_devices; ++i) {
		printf("  %s\n", getDeviceName(devices_id[i]).c_str());
	}

        size_t paramValueSize;
        clGetDeviceInfo(devices_id[0], CL_DEVICE_EXTENSIONS, 0, NULL, &paramValueSize);
        char * info = (char *)malloc(sizeof(char)*paramValueSize);
        clGetDeviceInfo(devices_id[0], CL_DEVICE_EXTENSIONS, paramValueSize, info, NULL);
        std::cout << "CL_DEVICE_EXTENSIONS:\t" << info << std::endl;

	context = clCreateContext(NULL, ret_num_devices, &devices_id[0], NULL, NULL, &ret);
	checkError(ret, "Failed to create context");
	command_queue = clCreateCommandQueue(context, devices_id[0], 0, &ret);

	std::string binary_file = getBoardBinaryFile("Conv2D", devices_id[0]);
	printf("Using Aocx: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &devices_id[0], 1);
	ret = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(ret, "Failed to build program");
	kernel = clCreateKernel(program, "Conv2D", &ret);
	checkError(ret, "Failed to create kernel");

	data_in = clCreateBuffer(context, CL_MEM_READ_WRITE, (image_width + image_pad_width)*(image_height + image_pad_height)*image_depth * sizeof(float), NULL, &ret);
	data_out = clCreateBuffer(context, CL_MEM_READ_WRITE, image_new_width*image_new_height*filter_kernel_depth * sizeof(float), NULL, &ret);
	filter_in = clCreateBuffer(context, CL_MEM_READ_WRITE, filter_kernel_width*filter_kernel_height*filter_kernel_depth * sizeof(float), NULL, &ret);

	//write image data into data_in buffer
	ret = clEnqueueWriteBuffer(command_queue, data_in, CL_TRUE, 0, (image_width + image_pad_width)*(image_height + image_pad_height)*image_depth * sizeof(float), data_padded, 0, NULL, NULL);
	checkError(ret, "write image data into device failed");

	//write filter data into filter_in buffer
	ret = clEnqueueWriteBuffer(command_queue, filter_in, CL_TRUE, 0, filter_kernel_width*filter_kernel_height*filter_kernel_depth * sizeof(float), filter_coe, 0, NULL, NULL);
	checkError(ret, "write filter data into device failed");

	//set kernel arguments
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&data_in);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_in);
	ret = clSetKernelArg(kernel, 2, sizeof(int), (void *)&filter_kernel_width);
	ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&filter_kernel_height);
	ret = clSetKernelArg(kernel, 4, sizeof(int), (void *)&filter_kernel_depth);
	ret = clSetKernelArg(kernel, 5, sizeof(int), (void *)&image_width);
	ret = clSetKernelArg(kernel, 6, sizeof(int), (void *)&image_height);
	ret = clSetKernelArg(kernel, 7, sizeof(int), (void *)&image_depth);
	ret = clSetKernelArg(kernel, 8, sizeof(int), (void *)&conv_width_step);
	ret = clSetKernelArg(kernel, 9, sizeof(int), (void *)&conv_height_step);
	ret = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&data_out);

	// set NDRangeKernel arguments
	work_dim = 3;
	global_item_size[0] = { image_new_width };
	global_item_size[1] = { image_new_height };
	global_item_size[2] = { filter_kernel_depth };
	local_item_size[0] = { 1 };
	local_item_size[1] = { 1 };
	local_item_size[2] = { 1 };

	//execute data parallel kernel */
	ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL,
			global_item_size, local_item_size, 0, NULL, NULL);

	// read data_out to host
	ret = clEnqueueReadBuffer(command_queue, data_out, CL_TRUE, 0,
			image_new_width*image_new_height*filter_kernel_depth * sizeof(float), result, 0, NULL, NULL);

	//display output data
        printf("\n");
	printf("Array data_out: \n");
        for (i = 0; i < filter_kernel_depth; i++)
        {
            printf("data_out depth[%d]:\n", i);
            for(j = 0; j < image_new_height; j++)
            {
                printf("row[%d]:\t", j);
                for(z = 0; z < image_new_width; z++)
                {
                    printf("%f, \t", result[i*image_new_height*image_new_width + j*image_new_width + z]);
                }
		printf("\n");
            }
	    printf("\n");
        }
	printf("\n");

	/*=================================================
	  release all opencl objects
	  =================================================*/
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(data_in);
	ret = clReleaseMemObject(data_out);
	ret = clReleaseMemObject(filter_in);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(result);
	free(kernel_str);

	return 0;
}
