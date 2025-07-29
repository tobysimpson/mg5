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
        
        lvl->vxl.n[0] = lvl->ele.n[0] + 1;
        lvl->vxl.n[1] = lvl->ele.n[1] + 1;
        lvl->vxl.n[2] = lvl->ele.n[2] + 1;
        
        lvl->vxl.i[0] = lvl->vxl.n[0] - 2;
        lvl->vxl.i[1] = lvl->vxl.n[1] - 2;
        lvl->vxl.i[2] = lvl->vxl.n[2] - 2;
        
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
        
    } //cycle
    
    return;
}


/*
 =====================
 norms
 =====================
 */

/*

//norms
void mg_nrm(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lvl)
{
    //arg
    ocl->err = clSetKernelArg(mg->ele_rsq,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(mg->ele_rsq,  1, sizeof(cl_mem),            (void*)&lvl->rr);
    ocl->err = clSetKernelArg(mg->ele_rsq,  2, sizeof(cl_mem),            (void*)&lvl->ee);
    
    //res sq
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->ele_rsq, 3, NULL, lvl->msh.ne_sz, NULL, 0, NULL, NULL);
    float r = mg_red(ocl, mg, lvl->ee, lvl->msh.ne_tot);
    
    //arg
    ocl->err = clSetKernelArg(mg->ele_esq,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(mg->ele_esq,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(mg->ele_esq,  2, sizeof(cl_mem),            (void*)&lvl->aa);
    ocl->err = clSetKernelArg(mg->ele_esq,  3, sizeof(cl_mem),            (void*)&lvl->ee);
    
    //err sq
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->ele_esq, 3, NULL, lvl->msh.ne_sz, NULL, 0, NULL, &ocl->event);
    float e = mg_red(ocl, mg, lvl->ee, lvl->msh.ne_tot);
    
    float dx3 = lvl->msh.dx*lvl->msh.dx2;
    
    //norms
//    printf("nrm [%2u,%2u,%2u] %+e %+e\n", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, sqrt(dx3*r), sqrt(dx3*e));
    printf("%10d %+e %+e\n", lvl->msh.nv_tot, sqrt(dx3*r), sqrt(dx3*e));
    
    return;
}
 
 */

/*

//fold (max 1024ˆ3)
float mg_red(struct ocl_obj *ocl, struct mg_obj *mg, cl_mem uu, cl_int n)
{
    //args
    ocl->err = clSetKernelArg(mg->vec_sum, 0, sizeof(cl_mem), (void*)&uu);
    ocl->err = clSetKernelArg(mg->vec_sum, 1, sizeof(cl_int), (void*)&n);

    uint l = ceil(log2(n));
    
    //loop
    for(int i=0; i<l; i++)
    {
        size_t p = pow(2,l-i-1);
        
//        printf("%2d %2d %u %zu\n", i, l, n, p);
    
        //calc
        ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->vec_sum, 1, NULL, &p, NULL, 0, NULL, NULL);
    }
    
    //result
    float r;
    
    //read
    ocl->err = clEnqueueReadBuffer(ocl->command_queue, uu, CL_TRUE, 0, sizeof(float), &r, 0, NULL, NULL);

    return r;
}

 */

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
