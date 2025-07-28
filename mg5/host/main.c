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


//multigrid voxel/element with image datatype
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
    mg.le = (cl_int3){1,1,1};
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
    cl_kernel ele_ini = clCreateKernel(ocl.program, "ele_ini", &ocl.err);

    //arg
    ocl.err = clSetKernelArg(ele_ini, 0, sizeof(struct msh_obj), &lvl->msh);
    ocl.err = clSetKernelArg(ele_ini, 1, sizeof(cl_mem)        , &lvl->gg);
    ocl.err = clSetKernelArg(ele_ini, 2, sizeof(cl_mem)        , &lvl->uu);
    ocl.err = clSetKernelArg(ele_ini, 3, sizeof(cl_mem)        , &lvl->bb);
    ocl.err = clSetKernelArg(ele_ini, 4, sizeof(cl_mem)        , &lvl->rr);
    
    //ini
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ele_ini, 3, NULL, lvl->ele.n, NULL, 0, NULL, NULL);

    /*
     ====================
     geometry
     ====================
     */
    
    //geom
    for(int l=0; l<mg.nl; l++)
    {
//        struct lvl_obj *lvl = &mg.lvls[l];
//        mg_geo(&ocl, &mg, lvl);
        
//        //write
//        wrt_xmf(&ocl, lvl, l, 0);
//        wrt_img1(&ocl, lvl->gg, &lvl->ele, "gg", l, 0);
//        wrt_img1(&ocl, lvl->uu, &lvl->ele, "uu", l, 0);
//        wrt_img1(&ocl, lvl->bb, &lvl->ele, "bb", l, 0);
//        wrt_img1(&ocl, lvl->rr, &lvl->ele, "rr", l, 0);
    }
    
    /*
     ====================
     solve
     ====================
     */
    
    //solve
//    mg_jac(&ocl, &mg, &mg.ops[0], &mg.lvls[0], 100);
    
    //project
    for(int l=0; l<(mg.nl-1); l++)
    {
        //levels
        struct lvl_obj *lf = &mg.lvls[l];
        struct lvl_obj *lc = &mg.lvls[l+1];
        
        //args
        ocl.err = clSetKernelArg(mg.ele_prj,  0, sizeof(cl_mem),            (void*)&lf->gg);      //fine
        ocl.err = clSetKernelArg(mg.ele_prj,  1, sizeof(cl_mem),            (void*)&lc->uu);      //coarse
        ocl.err = clSetKernelArg(mg.ele_prj,  2, sizeof(cl_mem),            (void*)&lc->gg);      //coarse
        
        //project
        ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, mg.ele_prj, 3, NULL, lc->ele.n, NULL, 0, NULL, NULL);

    }
    
    
//    clEnqueueCopyImage(ocl.command_queue, lf.uu, lf.rr, <#const size_t *#>, <#const size_t *#>, <#const size_t *#>, <#cl_uint#>, <#const cl_event *#>, <#cl_event *#>)
//    clEnqueueFillImage(<#cl_command_queue#>, <#cl_mem#>, <#const void *#>, <#const size_t *#>, <#const size_t *#>, <#cl_uint#>, <#const cl_event *#>, <#cl_event *#>)
    
    size_t ogn[3] = {0,0,0};
    cl_float4 ptn = {0.0f, 0.0f, 0.0f, 0.0f};
    clEnqueueFillImage(ocl.command_queue, mg.lvls[0].uu, &ptn, mg.ogn, mg.lvls[0].ele.n, 0, NULL, NULL);
    clFlush(ocl.command_queue);
    
    //copy
//    ocl.err = clEnqueueCopyBuffer(ocl.command_queue, lf.uu, pp, 0, 0, msh.ne_tot*sizeof(cl_float), 0, NULL, &ocl.event);
  
    //fill
//    cl_float4 ptn = {0e0f,0e0f,0e0f,0e0f};
//    ocl.err = clEnqueueFillBuffer(ocl.command_queue, vc, &ptn, sizeof(ptn), 0, msh.nv_tot*sizeof(ptn), 0, NULL, &ocl.event);
    
    
    //write all
    for(int l=0; l<mg.nl; l++)
    {
        struct lvl_obj *lvl = &mg.lvls[l];

        //write
        wrt_xmf(&ocl, lvl, l, 0);
        wrt_img1(&ocl, lvl->gg, &lvl->ele, "gg", l, 0);
        wrt_img1(&ocl, lvl->uu, &lvl->ele, "uu", l, 0);
        wrt_img1(&ocl, lvl->bb, &lvl->ele, "bb", l, 0);
        wrt_img1(&ocl, lvl->rr, &lvl->ele, "rr", l, 0);
    }
    
    
    

//    //write fine
//    wrt_xmf(&ocl, lvl, 0, 0);
//    wrt_img1(&ocl, lvl->gg, &lvl->ele, "gg", 0, 0);
//    wrt_img1(&ocl, lvl->uu, &lvl->ele, "uu", 0, 0);
//    wrt_img1(&ocl, lvl->bb, &lvl->ele, "bb", 0, 0);
//    wrt_img1(&ocl, lvl->rr, &lvl->ele, "rr", 0, 0);
    
    /*
     ====================
     clean
     ====================
     */
    
    //clean
    ocl.err = clReleaseKernel(ele_ini);

    mg_fin(&ocl, &mg);
    ocl_fin(&ocl);
    
    clock_gettime(CLOCK_REALTIME, &t1);

    printf("%f\n", (1e9f*(t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec))*1e-9);
    
    printf("done\n");
    
    return 0;
}
