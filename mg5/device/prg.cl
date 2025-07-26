//
//  prg.cl
//  fsi2
//
//  Created by toby on 29.05.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//


//#include "utl.h"
//#include "sdf.h"
//#include "mem.h"
//#include "cal.h"
//#include "lin.h"


/*
 ===================================
 mesh
 ===================================
 */

//object
struct msh_obj
{
    int3    le;
    int3    ne;
    int3    nv;
    
    int     ne_tot;
    int     nv_tot;
    
    float   dt;
    float   dx;
    float   dx2;
    float   rdx;
    float   rdx2;
    
    ulong   nv_sz[3];
    ulong   ne_sz[3];
    ulong   iv_sz[3];
    ulong   ie_sz[3];
};


/*
 ============================
 util
 ============================
 */

//global index
int utl_idx1(int4 pos, int3 dim)
{
    return pos.x + dim.x*pos.y + dim.x*dim.y*pos.z;
}

//local index 3x3x3
int utl_idx3(int4 pos)
{
    return pos.x + 3*pos.y + 9*pos.z;
}

/*
 ============================
 image test
 ============================
 */


//image test
kernel void test1(const struct msh_obj  msh,
                  write_only image3d_t img1,
                  write_only image3d_t img2)
{
    int4 off = {get_global_offset(0), get_global_offset(1), get_global_offset(2), 0};
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    
    printf("%v4d %v4d\n", off, pos);
    
    write_imagef(img1, pos, 1);
    write_imagef(img2, pos, 0);

    return;
}


//image test
kernel void test2(const struct msh_obj  msh,
                  read_only image3d_t img1,
                  read_only image3d_t img2)
{
    int4 off = {get_global_offset(0), get_global_offset(1), get_global_offset(2), 0};
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    
    printf("%v4d %v4d\n", off, pos);

    return;
}


