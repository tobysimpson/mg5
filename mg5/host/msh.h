//
//  msh.h
//  fsi2
//
//  Created by Toby Simpson on 14.04.2025.
//

#ifndef msh_h
#define msh_h

#include <stdio.h>
#include "ocl.h"


//object
struct msh_obj
{
    cl_float        dt;
    cl_float        dx;
    cl_float        dx2;
    cl_float        rdx2;
};


//init
void msh_ini(struct msh_obj *msh);

#endif /* msh_h */
