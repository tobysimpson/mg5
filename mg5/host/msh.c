//
//  msh.c
//  mg5
//
//  Created by Toby Simpson on 26/07/2025.
//

#include "msh.h"

//init
void msh_ini(struct msh_obj *msh)
{
    msh->ne         = (cl_int3){1<<msh->le.x, 1<<msh->le.y, 1<<msh->le.z};
    msh->nv         = (cl_int3){msh->ne.x+1, msh->ne.y+1, msh->ne.z+1};
    
    msh->ne_tot     = msh->ne.x*msh->ne.y*msh->ne.z;
    msh->nv_tot     = msh->nv.x*msh->nv.y*msh->nv.z;
    
    msh->dx2        = msh->dx*msh->dx;
    msh->rdx        = 1e0f/msh->dx;
    msh->rdx2       = 1e0f/msh->dx2;
    
    msh->nv_sz[0]   = msh->nv.x;
    msh->nv_sz[1]   = msh->nv.y;
    msh->nv_sz[2]   = msh->nv.z;
    
    msh->ne_sz[0]   = msh->ne.x;
    msh->ne_sz[1]   = msh->ne.y;
    msh->ne_sz[2]   = msh->ne.z;
    
    msh->iv_sz[0]   = msh->nv.x - 2;
    msh->iv_sz[1]   = msh->nv.y - 2;
    msh->iv_sz[2]   = msh->nv.z - 2;
    
    msh->ie_sz[0]   = msh->ne.x - 2;
    msh->ie_sz[1]   = msh->ne.y - 2;
    msh->ie_sz[2]   = msh->ne.z - 2;
    
    msh->of_sz[0]   = 1;
    msh->of_sz[1]   = 1;
    msh->of_sz[2]   = 1;
    
    printf("msh [%2u %2u %2u] [%4u %4u %4u] [%4u %4u %4u] %12u %12u %e\n",
           msh->le.x, msh->le.y, msh->le.z,
           msh->ne.x, msh->ne.y, msh->ne.z,
           msh->nv.x, msh->nv.y, msh->nv.z,
           msh->ne_tot,
           msh->nv_tot,
           msh->dx);

    return;
}



//    clGetImageInfo(img1, CL_IMAGE_ROW_PITCH,   sizeof(size_t), &dsc1.image_row_pitch,   NULL);
//    clGetImageInfo(img1, CL_IMAGE_SLICE_PITCH, sizeof(size_t), &dsc1.image_slice_pitch, NULL);
    
       
