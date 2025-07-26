//
//  main.c
//  fsi2
//
//  Created by Toby Simpson on 06.08.2024.
//


//#ifdef __APPLE__
//#include <OpenCL/opencl.h>
//#else
//#include <CL/cl.h>
//#endif

#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

#include "ocl.h"
#include "msh.h"
#include "mg.h"
#include "io.h"


//multigrid image test
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
    //timer
    struct timespec t0;
    struct timespec t1;
    
    clock_gettime(CLOCK_REALTIME, &t0);
    
    //create folders
    mkdir("/Users/toby/Downloads/raw", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir("/Users/toby/Downloads/xmf", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    /*
     ====================
     init
     ====================
     */
    
    //opencl
    struct ocl_obj ocl;
    ocl_ini(&ocl);
    
    //mesh
    struct msh_obj msh;
    msh.le = (cl_int3){2,2,2};
    msh.dx = 16e0f*powf(2e0f, -msh.le.x);
    msh.dt = 0.5f;
    msh_ini(&msh);
    
    //multigrid
    struct mg_obj mg;
    mg.nl = msh.le.x;
    mg_ini(&ocl, &mg, &msh);
    
    /*
     ====================
     memory
     ====================
     */
    

    struct lvl_obj *lvl = &mg.lvls[0];

    //description
    cl_image_format fmt1 = {CL_R, CL_FLOAT};
    cl_image_desc   dsc1 = {CL_MEM_OBJECT_IMAGE3D, msh.nv.x, msh.nv.y, msh.nv.z};

    //allocate
    cl_mem img1 = clCreateImage(ocl.context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl.err);
    cl_mem img2 = clCreateImage(ocl.context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl.err);
    


//    clGetImageInfo(img1, CL_IMAGE_ROW_PITCH,   sizeof(size_t), &dsc1.image_row_pitch,   NULL);
//    clGetImageInfo(img1, CL_IMAGE_SLICE_PITCH, sizeof(size_t), &dsc1.image_slice_pitch, NULL);
    
                      
                      
    //kernel
    cl_kernel test1 = clCreateKernel(ocl.program, "test1", &ocl.err);
    cl_kernel test2 = clCreateKernel(ocl.program, "test2", &ocl.err);

    //arg
    ocl.err = clSetKernelArg(test1, 0, sizeof(struct msh_obj), &msh);
    ocl.err = clSetKernelArg(test1, 1, sizeof(cl_mem)        , &img1);
    ocl.err = clSetKernelArg(test1, 2, sizeof(cl_mem)        , &img2);
    
    ocl.err = clSetKernelArg(test2, 0, sizeof(struct msh_obj), &msh);
    ocl.err = clSetKernelArg(test2, 1, sizeof(cl_mem)        , &img1);
    ocl.err = clSetKernelArg(test2, 2, sizeof(cl_mem)        , &img2);

    //run
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, test1, 3, NULL, lvl->msh.nv_sz, NULL, 0, NULL, NULL);
//    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, test2, 3, NULL, lvl->msh.nv_sz, NULL, 0, NULL, NULL);
    
    //write
    wrt_xmf(&ocl, &msh, 0, 0);
    wrt_img1(&ocl, img1, &dsc1, "uu", 0, 0);
    
    //clean
    ocl.err = clReleaseKernel(test1);
    ocl.err = clReleaseKernel(test2);
    
    ocl.err = clReleaseMemObject(img1);
    ocl.err = clReleaseMemObject(img2);

    
    //clean
    mg_fin(&ocl, &mg);
    ocl_fin(&ocl);
    
    clock_gettime(CLOCK_REALTIME, &t1);

    printf("%d %f\n", msh.nv_tot, (1e9f*(t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec))*1e-9);
    
    printf("done\n");
    
    return 0;
}
