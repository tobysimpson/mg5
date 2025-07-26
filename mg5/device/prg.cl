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
    float        dt;
    float        dx;
    float        dx2;
    float        rdx2;
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
 memory
 ============================
 */


void mem_rgs6(read_only image3d_t uu, float4 ss[6], int4 pos)
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
 sdf
 ============================
 */

//sphere
float sdf_sph(float4 x, float4 c, float r)
{
    return length(x.xyz - c.xyz) - r;
}


//cuboid
float sdf_cub(float4 x, float4 c, float4 r)
{
    float4 d = fabs(x - c) - r;
    
    return max(d.x,max(d.y,d.z));
}


/*
 ============================
 ini
 ============================
 */


//ini
kernel void vtx_ini(const struct msh_obj  msh,
                    write_only image3d_t  gg,
                    write_only image3d_t  uu,
                    write_only image3d_t  bb,
                    write_only image3d_t  rr)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
//    int4 dim = get_image_dim(uu);
    
    float4 x = msh.dx*convert_float4(pos);
    
    write_imagef(gg, pos, 1.0f);
    write_imagef(uu, pos, sin(x.x));
    write_imagef(bb, pos, sin(x.x));
    write_imagef(rr, pos, 0);

    return;
}


//geom
kernel void vtx_geo(const       struct msh_obj  msh,
                    write_only  image3d_t       gg)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    
    float4 x = msh.dx*convert_float4(pos);
    
    float g1 = sdf_sph(x,(float4){0.6f, 0.6f, 0.4f, 0.0f}, 0.25f);
    float g2 = sdf_cub(x,(float4){0.4f, 0.4f, 0.6f, 0.0f}, (float4){0.25f, 0.25f, 0.25f, 0.0f});
    float g = min(g1,g2);
    
    write_imagef(gg, pos, g);

    return;
}

/*
 ============================
 solve
 ============================
 */

//image test
kernel void vtx_fwd(const struct msh_obj  msh,
                    read_only     image3d_t       uu,
                    write_only    image3d_t       rr)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
//    int4 dim = get_image_dim(img1);
    
    printf("%v4d\n", pos);
    
    //stencil
    float4 ss[6];
    mem_rgs6(uu, ss, pos);
    
    //sum
    float s = 0e0f;
    
    for(int i=0; i<6; i++)
    {
        s += ss[i].x;
    }
    
    float4 u = read_imagef(uu, pos);
    write_imagef(rr, pos, s - 6.0f*u.x);

    return;
}

/*
 ============================
 transfer
 ============================
 */

/*

//project
kernel void vtx_prj(const  struct msh_obj   msh,    //coarse    (out)
                    global float            *rr,    //fine      (in)
                    global float            *uu,    //coarse    (out)
                    global float            *bb)    //coarse    (out)
{
    int4  vtx_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)} + 1; //interior
    int   vtx_idx0  = utl_idx(vtx_pos, msh.nv);   //coarse
    
    //fine
    int3 pos = 2*vtx_pos;
    int3 dim = 2*msh.ne+1;
    
    //injection
    int  vtx_idx1  = utl_idx1(pos,dim);
    
    //store/reset
    uu[vtx_idx0] = 0e0f;
    bb[vtx_idx0] = rr[vtx_idx1];

    return;
}


//interpolate
kernel void vtx_itp(const  struct msh_obj   msh,    //fine      (out)
                    global float            *u0,    //coarse    (in)
                    global float            *u1)    //fine      (out)
{
    int3  vtx_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)} + 1; //fine - interior
    int   vtx_idx  = utl_idx1(vtx_pos, msh.nv);   //fine
    
    //coarse
    float3 pos = convert_float3(vtx_pos)/2e0f;
    
    //round up/down
    int3 pos0 = convert_int3(floor(pos));
    int3 pos1 = convert_int3(ceil(pos));
    
    //coarse dims
    int3 dim = 1 + msh.ne/2;
    
    //sum
    float s = 0e0f;
    s += u0[utl_idx1((int3){pos0.x, pos0.y, pos0.z}, dim)];
    s += u0[utl_idx1((int3){pos1.x, pos0.y, pos0.z}, dim)];
    s += u0[utl_idx1((int3){pos0.x, pos1.y, pos0.z}, dim)];
    s += u0[utl_idx1((int3){pos1.x, pos1.y, pos0.z}, dim)];
    s += u0[utl_idx1((int3){pos0.x, pos0.y, pos1.z}, dim)];
    s += u0[utl_idx1((int3){pos1.x, pos0.y, pos1.z}, dim)];
    s += u0[utl_idx1((int3){pos0.x, pos1.y, pos1.z}, dim)];
    s += u0[utl_idx1((int3){pos1.x, pos1.y, pos1.z}, dim)];
    
    u1[vtx_idx] += s*0.125f;
    
    return;
}

*/
