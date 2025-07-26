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
int utl_idx(int4 pos, int4 dim)
{
    return pos.x + dim.x*pos.y + dim.x*dim.y*pos.z;
}

/*
 ============================
 stencil
 ============================
 */


void mem_ss(read_only image3d_t uu, float4 ss[6], int4 pos)
{
    ss[0] = read_imagef(uu, pos - (int4){1,0,0,0});
    ss[1] = read_imagef(uu, pos + (int4){1,0,0,0});
    ss[2] = read_imagef(uu, pos - (int4){0,1,0,0});
    ss[3] = read_imagef(uu, pos + (int4){0,1,0,0});
    ss[4] = read_imagef(uu, pos - (int4){0,0,1,0});
    ss[5] = read_imagef(uu, pos + (int4){0,0,1,0});

    return;
}


/*
 ============================
 image
 ============================
 */


//image test
kernel void test1(const struct msh_obj  msh,
                  write_only image3d_t  uu,
                  write_only image3d_t  rr)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    int4 dim = get_image_dim(uu);
    
//    printf("%v4d\n", pos);
    
    int idx = utl_idx(pos, dim);
    
    write_imagef(uu, pos, idx);
    write_imagef(rr, pos, 0);

    return;
}


//image test
kernel void test2(const         struct msh_obj  msh,
                  read_only     image3d_t       uu,
                  write_only    image3d_t       rr)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
//    int4 dim = get_image_dim(img1);
    
    printf("%v4d ", pos);
    
    float4 ss[6];
    mem_ss(uu, ss, pos);
    
    for(int i=0; i<6; i++)
    {
        printf("%f ", ss[i].x);
    }
    printf("\n");
    
    float4 u = read_imagef(uu, pos);
    write_imagef(rr, pos, u.x);

    return;
}


