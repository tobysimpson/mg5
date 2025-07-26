//
//  mg.c
//  fsi2
//
//  Created by Toby Simpson on 14.04.2025.
//
#include <time.h>
#include <math.h>
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
        
        lvl->vtx.n[0] = lvl->ele.n[0] + 1;
        lvl->vtx.n[1] = lvl->ele.n[1] + 1;
        lvl->vtx.n[2] = lvl->ele.n[2] + 1;
        
        //dx
        lvl->msh.dx = mg->dx*powf(2e0f,l);
        lvl->msh.dt = mg->dt;
        
        //mesh
        msh_ini(&lvl->msh);
        
        //description
        cl_image_format fmt1 = {CL_R, CL_FLOAT};
        cl_image_desc   dsc1 = {CL_MEM_OBJECT_IMAGE3D, lvl->vtx.n[0], lvl->vtx.n[1], lvl->vtx.n[2]};

        //allocate
        lvl->gg = clCreateImage(ocl->context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl->err);
        lvl->uu = clCreateImage(ocl->context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl->err);
        lvl->bb = clCreateImage(ocl->context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl->err);
        lvl->rr = clCreateImage(ocl->context, CL_MEM_HOST_READ_ONLY, &fmt1, &dsc1, NULL, &ocl->err);
    }
    
    //geo
    mg->vtx_geo = clCreateKernel(ocl->program, "vtx_geo", &ocl->err);
    
    //trans
    mg->vtx_prj = clCreateKernel(ocl->program, "vtx_prj", &ocl->err);
    mg->vtx_itp = clCreateKernel(ocl->program, "vtx_itp", &ocl->err);
    
    //mech
    mg->ops[0].vtx_res = clCreateKernel(ocl->program, "vtx_res", &ocl->err);
    mg->ops[0].vtx_jac = clCreateKernel(ocl->program, "vtx_jac", &ocl->err);
    
    //offset
    mg->off[0] = 1;
    mg->off[1] = 1;
    mg->off[2] = 1;
    
    return;
}



//geometry
void mg_geo(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lvl, cl_mem *ss)
{
    //args
    ocl->err = clSetKernelArg(mg->vtx_geo,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(mg->vtx_geo,  1, sizeof(cl_mem),            (void*)&lvl->gg);
    
    //geo
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->vtx_geo, 3, NULL, (size_t*)lvl->vtx.n, NULL, 0, NULL, &ocl->event);

    return;
}


//jacobi (no res)
void mg_jac(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl, int nj)
{
    //args
    ocl->err = clSetKernelArg(op->vtx_res,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(op->vtx_res,  1, sizeof(cl_mem),            (void*)&lvl->gg);
    ocl->err = clSetKernelArg(op->vtx_res,  2, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(op->vtx_res,  3, sizeof(cl_mem),            (void*)&lvl->bb);
    ocl->err = clSetKernelArg(op->vtx_res,  4, sizeof(cl_mem),            (void*)&lvl->rr);
    
    //jac
    ocl->err = clSetKernelArg(op->vtx_jac,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(op->vtx_jac,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(op->vtx_jac,  2, sizeof(cl_mem),            (void*)&lvl->rr);
    
    //smooth
    for(int j=0; j<nj; j++)
    {
        ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vtx_res, 3, mg->off, lvl->vtx.i, NULL, 0, NULL, NULL);
        ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vtx_jac, 3, mg->off, lvl->vtx.i, NULL, 0, NULL, NULL);
    }

    return;
}


//residual
void mg_res(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl)
{
    //args
    ocl->err = clSetKernelArg(op->vtx_res,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(op->vtx_res,  1, sizeof(cl_mem),            (void*)&lvl->gg);
    ocl->err = clSetKernelArg(op->vtx_res,  2, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(op->vtx_res,  3, sizeof(cl_mem),            (void*)&lvl->bb);
    ocl->err = clSetKernelArg(op->vtx_res,  4, sizeof(cl_mem),            (void*)&lvl->rr);
    
    //residual
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vtx_res, 3, mg->off, lvl->vtx.i, NULL, 0, NULL, NULL);

    return;
}


//interp
void mg_itp(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc)
{
    //args
    ocl->err = clSetKernelArg(mg->vtx_itp,  0, sizeof(struct msh_obj),    (void*)&lf->msh);     //fine
    ocl->err = clSetKernelArg(mg->vtx_itp,  1, sizeof(cl_mem),            (void*)&lc->uu);      //coarse
    ocl->err = clSetKernelArg(mg->vtx_itp,  2, sizeof(cl_mem),            (void*)&lf->uu);      //fine
    
    //interp
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->vtx_itp, 3, mg->off, lf->vtx.i, NULL, 0, NULL, NULL);
    
    return;
}


//project
void mg_prj(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc)
{
    //args
    ocl->err = clSetKernelArg(mg->vtx_prj,  0, sizeof(struct msh_obj),    (void*)&lc->msh);     //coarse
    ocl->err = clSetKernelArg(mg->vtx_prj,  1, sizeof(cl_mem),            (void*)&lf->rr);      //fine
    ocl->err = clSetKernelArg(mg->vtx_prj,  2, sizeof(cl_mem),            (void*)&lc->uu);      //coarse
    ocl->err = clSetKernelArg(mg->vtx_prj,  3, sizeof(cl_mem),            (void*)&lc->bb);      //coarse
    
    //project
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->vtx_prj, 3, NULL, lc->vtx.i, NULL, 0, NULL, NULL);
    
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
            
        } //dsc
        
    } //cycle
    
    return;
}


//final
void mg_fin(struct ocl_obj *ocl, struct mg_obj *mg)
{
    ocl->err = clReleaseKernel(mg->vtx_geo);
    ocl->err = clReleaseKernel(mg->vtx_prj);
    ocl->err = clReleaseKernel(mg->vtx_itp);
    
    ocl->err = clReleaseKernel(mg->ops[0].vtx_res);
    ocl->err = clReleaseKernel(mg->ops[0].vtx_jac);

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
