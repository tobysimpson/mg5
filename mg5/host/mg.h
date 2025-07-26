//
//  mg.h
//  mg1
//
//  Created by toby on 10.06.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef mg_h
#define mg_h

#include <math.h>
#include "msh.h"

//object
struct lvl_obj
{
    struct msh_obj  msh;
    
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
    cl_kernel       vtx_jac;
    cl_kernel       vtx_res;

};


//object
struct mg_obj
{
    //levels
    int             nl;     //levels

    //array
    struct lvl_obj *lvls;
    
    //kernels
    cl_kernel       ele_geo;
    cl_kernel       vtx_geo;
    cl_kernel       vtx_prj;
    cl_kernel       vtx_itp;
    
    //ops
    struct op_obj ops[1];
};



//init
void mg_ini(struct ocl_obj *ocl, struct mg_obj *mg, struct msh_obj *msh);
void mg_fin(struct ocl_obj *ocl, struct mg_obj *mg);

void mg_itp(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc);
void mg_prj(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc);

void mg_geo(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lvl, cl_mem *ss);

void mg_jac(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl, int nj);
void mg_res(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl);

void mg_cyc(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, int nl, int nj, int nc);


#endif /* mg_h */
