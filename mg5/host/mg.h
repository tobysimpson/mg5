//
//  mg.h
//  mg1
//
//  Created by toby on 10.06.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef mg_h
#define mg_h

#include <stdio.h>
#include <math.h>
#include "ocl.h"

//object
struct msh_obj
{
    cl_float        dt;
    cl_float        dx;
    cl_float        dx2;
    cl_float        rdx2;
};

//object
struct dim_obj
{
    size_t    n[3];   //all
    size_t    i[3];   //interior
};

//object
struct lvl_obj
{
    cl_int3         le;
    
    struct msh_obj  msh;
    
    struct dim_obj  vxl;
    struct dim_obj  ele;

    //memory
    cl_mem  uu;
    cl_mem  bb;
    cl_mem  rr;
    cl_mem  gg;
};


//object
struct op_obj
{
    //operator
    cl_kernel       vxl_fwd;
    cl_kernel       vxl_jac;
    cl_kernel       vxl_res;
};


//object
struct mg_obj
{
    //dims
    cl_int3     le;     //log2 dims
    cl_int      nl;     //levels
    cl_float    dx;
    cl_float    dt;

    //array
    struct lvl_obj *lvls;
    
    //kernels
    cl_kernel       vxl_geo;
    cl_kernel       vxl_prj;
    cl_kernel       vxl_itp;
    
    //ops
    struct op_obj ops[1];
    
    //origin, offset
    size_t ogn[3];
    size_t off[3];
};



//init
void mg_ini(struct ocl_obj *ocl, struct mg_obj *mg);
void mg_fin(struct ocl_obj *ocl, struct mg_obj *mg);

void mg_itp(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc);
void mg_prj(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc);

void mg_jac(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl, int nj);
void mg_res(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl);

void mg_cyc(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, int nl, int nj, int nc);


#endif /* mg_h */
