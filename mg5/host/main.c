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
    
    
    //multigrid
    struct mg_obj mg;
    mg.le = (cl_int3){2,2,2};
    mg.nl = mg.le.x;
    mg.dx = 1.0f;
    mg.dt = 0.5f;
    mg_ini(&ocl, &mg);
    
    /*
     ====================
     memory
     ====================
     */
    
    struct lvl_obj *lvl = &mg.lvls[0];

//    //description
//    cl_image_format fmt1 = {CL_R, CL_FLOAT};
//    cl_image_desc   dsc1 = {CL_MEM_OBJECT_IMAGE3D, msh.nv.x, msh.nv.y, msh.nv.z};
//
//    //allocate
//    cl_mem uu = clCreateImage(ocl.context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl.err);
//    cl_mem rr = clCreateImage(ocl.context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl.err);
                      
    //kernel
    cl_kernel vtx_ini = clCreateKernel(ocl.program, "test1", &ocl.err);

    //arg
    ocl.err = clSetKernelArg(vtx_ini, 0, sizeof(struct msh_obj), &lvl->msh);
    ocl.err = clSetKernelArg(vtx_ini, 1, sizeof(cl_mem)        , &lvl->uu);
    ocl.err = clSetKernelArg(vtx_ini, 2, sizeof(cl_mem)        , &lvl->rr);
    
    //ini
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, vtx_ini, 3, NULL, lvl->vtx.n, NULL, 0, NULL, NULL);

    
    //write
    wrt_xmf(&ocl, lvl, 0, 0);
    wrt_img1(&ocl, lvl->uu, &lvl->vtx, "gg", 0, 0);
    wrt_img1(&ocl, lvl->bb, &lvl->vtx, "uu", 0, 0);
    wrt_img1(&ocl, lvl->bb, &lvl->vtx, "rr", 0, 0);
    
    //clean
    ocl.err = clReleaseKernel(vtx_ini);

    mg_fin(&ocl, &mg);
    ocl_fin(&ocl);
    
    clock_gettime(CLOCK_REALTIME, &t1);

    printf("%f\n", (1e9f*(t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec))*1e-9);
    
    printf("done\n");
    
    return 0;
}
