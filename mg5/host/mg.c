//
//  mg.c
//  fsi2
//
//  Created by Toby Simpson on 14.04.2025.
//

#include "mg.h"


//init
void mg_ini(struct ocl_obj *ocl, struct mg_obj *mg)
{
    printf("mg [%d,%d,%d] %d\n",mg->le.x, mg->le.y, mg->le.z, mg->nl);
    
    //levels
    mg->lvls = malloc(mg->nl*sizeof(struct lvl_obj));
    
    //levels
    for(int l=0; l<mg->nl; l++)
    {
        //instance
        struct lvl_obj *lvl = &mg->lvls[l];
        
        //dims
        lvl->le = (cl_int3){mg->le.x-l, mg->le.y-l, mg->le.z-l};
        
        lvl->ele.n[0] = 1<<lvl->le.x;
        lvl->ele.n[1] = 1<<lvl->le.y;
        lvl->ele.n[2] = 1<<lvl->le.z;
        
        lvl->ele.i[0] = lvl->ele.n[0] - 2;
        lvl->ele.i[1] = lvl->ele.n[1] - 2;
        lvl->ele.i[2] = lvl->ele.n[2] - 2;
        
        lvl->ele.tot = lvl->ele.n[0]*lvl->ele.n[1]*lvl->ele.n[2];
        
        lvl->vxl.n[0] = lvl->ele.n[0] + 1;
        lvl->vxl.n[1] = lvl->ele.n[1] + 1;
        lvl->vxl.n[2] = lvl->ele.n[2] + 1;
        
        lvl->vxl.i[0] = lvl->vxl.n[0] - 2;
        lvl->vxl.i[1] = lvl->vxl.n[1] - 2;
        lvl->vxl.i[2] = lvl->vxl.n[2] - 2;
        
        lvl->vxl.tot = lvl->vxl.n[0]*lvl->vxl.n[1]*lvl->vxl.n[2];
        
        //dx
        lvl->msh.dx = mg->dx*powf(2e0f,l);
        lvl->msh.dt = mg->dt;
        
        //mesh
        lvl->msh.dx2        = lvl->msh.dx*lvl->msh.dx;
        lvl->msh.rdx2       = 1e0f/lvl->msh.dx2;
        
        printf("lvl %d [%2d,%2d,%2d] [%4zu,%4zu,%4zu] %10zu %f %f\n", l,
               lvl->le.x,
               lvl->le.y,
               lvl->le.z,
               lvl->vxl.n[0],
               lvl->vxl.n[1],
               lvl->vxl.n[2],
               lvl->vxl.n[0]*lvl->vxl.n[1]*lvl->vxl.n[2],
               lvl->msh.dx,
               lvl->msh.dt);

        //description
        cl_image_format fmt1 = {CL_R, CL_FLOAT};
        cl_image_desc   dsc1 = {CL_MEM_OBJECT_IMAGE3D, lvl->vxl.n[0], lvl->vxl.n[1], lvl->vxl.n[2]};

        //allocate
        lvl->gg = clCreateImage(ocl->context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl->err);
        lvl->uu = clCreateImage(ocl->context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl->err);
        lvl->bb = clCreateImage(ocl->context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl->err);
        lvl->rr = clCreateImage(ocl->context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl->err);
    }
    
    //trans
    mg->vxl_prj = clCreateKernel(ocl->program, "vxl_prj", &ocl->err);
    mg->vxl_itp = clCreateKernel(ocl->program, "vxl_itp", &ocl->err);
    
    //poisson
    mg->ops[0].vxl_res = clCreateKernel(ocl->program, "vxl_res", &ocl->err);
    mg->ops[0].vxl_jac = clCreateKernel(ocl->program, "vxl_jac", &ocl->err);
    
    //origin
    mg->ogn[0] = 0;
    mg->ogn[1] = 0;
    mg->ogn[2] = 0;
    
    //offset
    mg->off[0] = 1;
    mg->off[1] = 1;
    mg->off[2] = 1;
    
    return;
}


//jacobi (twice)
void mg_jac(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl, int nj)
{
    //jac
    ocl->err = clSetKernelArg(op->vxl_jac,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(op->vxl_jac,  1, sizeof(cl_mem),            (void*)&lvl->gg);
    ocl->err = clSetKernelArg(op->vxl_jac,  2, sizeof(cl_mem),            (void*)&lvl->bb);
    
    //smooth/swap
    for(int j=0; j<nj; j++)
    {
        ocl->err = clSetKernelArg(op->vxl_jac,  3, sizeof(cl_mem),            (void*)&lvl->uu);
        ocl->err = clSetKernelArg(op->vxl_jac,  4, sizeof(cl_mem),            (void*)&lvl->rr);
        
        ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vxl_jac, 3, mg->ogn, lvl->vxl.n, NULL, 0, NULL, NULL);
        
        ocl->err = clSetKernelArg(op->vxl_jac,  3, sizeof(cl_mem),            (void*)&lvl->rr);
        ocl->err = clSetKernelArg(op->vxl_jac,  4, sizeof(cl_mem),            (void*)&lvl->uu);
        
        ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vxl_jac, 3, mg->ogn, lvl->vxl.n, NULL, 0, NULL, NULL);
    }

    return;
}


//residual
void mg_res(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl)
{
    //args
    ocl->err = clSetKernelArg(op->vxl_res,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(op->vxl_res,  1, sizeof(cl_mem),            (void*)&lvl->gg);
    ocl->err = clSetKernelArg(op->vxl_res,  2, sizeof(cl_mem),            (void*)&lvl->bb);
    ocl->err = clSetKernelArg(op->vxl_res,  3, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(op->vxl_res,  4, sizeof(cl_mem),            (void*)&lvl->rr);
    
    //residual
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vxl_res, 3, mg->ogn, lvl->vxl.n, NULL, 0, NULL, NULL);

    return;
}


//project
void mg_prj(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc)
{
    //args
    ocl->err = clSetKernelArg(mg->vxl_prj,  0, sizeof(cl_mem),            (void*)&lf->rr);      //fine
    ocl->err = clSetKernelArg(mg->vxl_prj,  1, sizeof(cl_mem),            (void*)&lc->bb);      //coarse

    //project
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->vxl_prj, 3, mg->ogn, lc->vxl.n, NULL, 0, NULL, NULL);
    
    //store soln
    ocl->err = clEnqueueCopyImage(ocl->command_queue, lf->uu, lf->rr, mg->ogn, mg->ogn, lf->vxl.n, 0, NULL, NULL);
    
    //clear coarse soln
    cl_float4 ptn = {0.0f, 0.0f, 0.0f, 0.0f};
    clEnqueueFillImage(ocl->command_queue, lc->uu, &ptn, mg->ogn, lc->vxl.n, 0, NULL, NULL);
    
    return;
}


//interp
void mg_itp(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc)
{
    //args
    ocl->err = clSetKernelArg(mg->vxl_itp,  0, sizeof(cl_mem),            (void*)&lc->uu);      //coarse
    ocl->err = clSetKernelArg(mg->vxl_itp,  1, sizeof(cl_mem),            (void*)&lf->uu);      //fine
    ocl->err = clSetKernelArg(mg->vxl_itp,  2, sizeof(cl_mem),            (void*)&lf->rr);      //fine (soln stored)

    //interp
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->vxl_itp, 3, mg->ogn, lf->vxl.n, NULL, 0, NULL, NULL);

    return;
}



//v-cycles
void mg_cyc(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, int nl, int nj, int nc)
{
    //cycle
    for(int c=0; c<nc; c++)
    {
        //descend
        for(int l=0; l<(nl-1); l++)
        {
            //levels
            struct lvl_obj *lf = &mg->lvls[l];
            struct lvl_obj *lc = &mg->lvls[l+1];
            
            //pre
            mg_jac(ocl, mg, op, lf, nj);
            
            //res
            mg_res(ocl, mg, op, lf);
            
            //prj
            mg_prj(ocl, mg, lf, lc);
            
        } //dsc
        
        //coarse
        mg_jac(ocl, mg, op, &mg->lvls[nl-1], nj);
        
        //ascend
        for(int l=(nl-2); l>=0; l--)
        {
            //levels
            struct lvl_obj *lf = &mg->lvls[l];
            struct lvl_obj *lc = &mg->lvls[l+1];
            
            //itp
            mg_itp(ocl, mg, lf, lc);
           
            //post
            mg_jac(ocl, mg, op, lf, nj);
            
        } //asc
        
        //res
        mg_res(ocl, mg, op, &mg->lvls[0]);
        
        //sum
        double s = img_sum(ocl, mg->lvls[0].rr, 1.0);           //sum
        double v = 1.0/(double)mg->lvls[0].vxl.tot;             //volume
        
        printf("%e\n",sqrt(s*v));
        
    } //cycle
    
    return;
}


/*
 =====================
 norms
 =====================
 */

//p-sum
double img_sum(struct ocl_obj *ocl, cl_mem img, double p)
{
    float* ptr;
    
    size_t ogn[3] = {0,0,0};
    size_t dim[3];
    size_t rp;
    size_t sp;
    
    ocl->err = clGetImageInfo(img, CL_IMAGE_WIDTH,          sizeof(size_t), &dim[0],    NULL);
    ocl->err = clGetImageInfo(img, CL_IMAGE_HEIGHT,         sizeof(size_t), &dim[1],    NULL);
    ocl->err = clGetImageInfo(img, CL_IMAGE_DEPTH,          sizeof(size_t), &dim[2],    NULL);
    ocl->err = clGetImageInfo(img, CL_IMAGE_ROW_PITCH,      sizeof(size_t), &rp,        NULL);
    ocl->err = clGetImageInfo(img, CL_IMAGE_SLICE_PITCH,    sizeof(size_t), &sp,        NULL);
    
    size_t tot = dim[0]*dim[1]*dim[2];
    
    //map
    ptr = clEnqueueMapImage(ocl->command_queue, img, CL_TRUE, CL_MAP_READ, ogn, dim, &rp, &sp, 0, NULL, NULL, NULL);
    
    //total
    double s = 0.0f;
    
    //sum
    for(int i=0; i<tot; i++)
    {
        s += pow((double)ptr[i], p);
    }
    
    //unmap
    clEnqueueUnmapMemObject(ocl->command_queue, img, ptr, 0, NULL, NULL);

    return s;
}

/*
 =====================
 clean
 =====================
 */

//final
void mg_fin(struct ocl_obj *ocl, struct mg_obj *mg)
{
    ocl->err = clReleaseKernel(mg->vxl_prj);
    ocl->err = clReleaseKernel(mg->vxl_itp);
    
    ocl->err = clReleaseKernel(mg->ops[0].vxl_res);
    ocl->err = clReleaseKernel(mg->ops[0].vxl_jac);

    //levels
    for(int l=0; l<mg->nl; l++)
    {
        //device
        ocl->err = clReleaseMemObject(mg->lvls[l].uu);
        ocl->err = clReleaseMemObject(mg->lvls[l].bb);
        ocl->err = clReleaseMemObject(mg->lvls[l].rr);
        ocl->err = clReleaseMemObject(mg->lvls[l].gg);
    }
    
    //mem
    free(mg->lvls);

    return;
}
