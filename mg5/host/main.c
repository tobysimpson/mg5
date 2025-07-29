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
#include "mg.h"
#include "io.h"


//multigrid voxel/image
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
    mg.le = (cl_int3){4,4,4};
    mg.nl = mg.le.x;
    mg.dx = 2.0f*powf(2e0f, -mg.le.x);  //[-1,+1]
    mg.dt = 0.5f;
    mg_ini(&ocl, &mg);
    
    /*
     ====================
     memory
     ====================
     */
    
    struct lvl_obj *lvl = &mg.lvls[0];
                      
    //kernel
    cl_kernel vxl_ini = clCreateKernel(ocl.program, "vxl_ini", &ocl.err);

    //arg
    ocl.err = clSetKernelArg(vxl_ini, 0, sizeof(struct msh_obj), &lvl->msh);
    ocl.err = clSetKernelArg(vxl_ini, 1, sizeof(cl_mem)        , &lvl->gg);
    ocl.err = clSetKernelArg(vxl_ini, 2, sizeof(cl_mem)        , &lvl->bb);
    ocl.err = clSetKernelArg(vxl_ini, 3, sizeof(cl_mem)        , &lvl->uu);
    ocl.err = clSetKernelArg(vxl_ini, 4, sizeof(cl_mem)        , &lvl->rr);
    
    //ini
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, vxl_ini, 3, NULL, lvl->vxl.n, NULL, 0, NULL, NULL);

    /*
     ====================
     geometry
     ====================
     */
    
    //project
    for(int l=0; l<(mg.nl-1); l++)
    {
        //levels
        struct lvl_obj *lf = &mg.lvls[l];
        struct lvl_obj *lc = &mg.lvls[l+1];

        //args
        ocl.err = clSetKernelArg(mg.vxl_prj,  0, sizeof(cl_mem),            (void*)&lf->gg);      //fine
        ocl.err = clSetKernelArg(mg.vxl_prj,  1, sizeof(cl_mem),            (void*)&lc->gg);      //coarse

        //project
        ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, mg.vxl_prj, 3, mg.ogn, lc->vxl.n, NULL, 0, NULL, NULL);
    }
    
    /*
     ====================
     solve
     ====================
     */
    
    //solve
    mg_jac(&ocl, &mg, &mg.ops[0], &mg.lvls[0], 100);
    
    //jac
//    ocl.err = clSetKernelArg(mg.ops[0].vxl_jac,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
//    ocl.err = clSetKernelArg(mg.ops[0].vxl_jac,  1, sizeof(cl_mem),            (void*)&lvl->gg);
//    ocl.err = clSetKernelArg(mg.ops[0].vxl_jac,  2, sizeof(cl_mem),            (void*)&lvl->bb);
//    ocl.err = clSetKernelArg(mg.ops[0].vxl_jac,  3, sizeof(cl_mem),            (void*)&lvl->uu);
//    ocl.err = clSetKernelArg(mg.ops[0].vxl_jac,  4, sizeof(cl_mem),            (void*)&lvl->rr);
//    
//    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, mg.ops[0].vxl_jac, 3, mg.ogn, lvl->vxl.n, NULL, 0, NULL, NULL);
    


    //write all
    for(int l=0; l<mg.nl; l++)
    {
        struct lvl_obj *lvl = &mg.lvls[l];

        //write
        wrt_xmf(&ocl, lvl, l, 0);
        wrt_img(&ocl, lvl->gg, &lvl->vxl, "gg", l, 0);
        wrt_img(&ocl, lvl->uu, &lvl->vxl, "uu", l, 0);
        wrt_img(&ocl, lvl->bb, &lvl->vxl, "bb", l, 0);
        wrt_img(&ocl, lvl->rr, &lvl->vxl, "rr", l, 0);
    }
    

//    //write fine
//    wrt_xmf(&ocl, lvl, 0, 0);
//    wrt_img1(&ocl, lvl->gg, &lvl->vxl, "gg", 0, 0);
//    wrt_img1(&ocl, lvl->uu, &lvl->vxl, "uu", 0, 0);
//    wrt_img1(&ocl, lvl->bb, &lvl->vxl, "bb", 0, 0);
//    wrt_img1(&ocl, lvl->rr, &lvl->vxl, "rr", 0, 0);
    
    /*
     ====================
     clean
     ====================
     */
    
    //clean
    ocl.err = clReleaseKernel(vxl_ini);

    mg_fin(&ocl, &mg);
    ocl_fin(&ocl);
    
    //timer
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("%f\n", (1e9f*(t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec))*1e-9);
    
    printf("done\n");
    
    return 0;
}
