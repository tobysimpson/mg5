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

//in-bounds
int utl_bnd(int4 pos, int4 dim)
{
    return all(pos.xyz>=0)*all(pos.xyz<dim.xyz);
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
kernel void ele_ini(const struct msh_obj  msh,
                    write_only image3d_t  gg,
                    write_only image3d_t  uu,
                    write_only image3d_t  bb,
                    write_only image3d_t  rr)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    int4 dim = get_image_dim(uu);
    
//    float4 x = msh.dx*convert_float4(pos);
    
    float4 u = 0e0f + (pos.x==0) - (pos.x==(dim.x-1));  //init
    
    write_imagef(gg, pos, 1.0f);
    write_imagef(uu, pos, u);
    write_imagef(bb, pos, 0.0f);
    write_imagef(rr, pos, u);           //bounds here too

    return;
}


//geom
kernel void ele_geo(const       struct msh_obj  msh,
                    write_only  image3d_t       gg)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    
    float4 x = msh.dx*(convert_float4(pos) + 0.5f);
    
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

//jacobi
kernel void ele_jac(const struct msh_obj  msh,
                    read_only     image3d_t       gg,
                    read_only     image3d_t       uu,
                    read_only     image3d_t       bb,
                    write_only    image3d_t       rr)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    
    //read
    float4 b = read_imagef(bb, pos);
    float4 g = read_imagef(gg, pos);
    
    if(g.x>0.0f)
    {
        //stencil
        float4 gg6[6];
        mem_rgs6(gg, gg6, pos);
        
        float4 uu6[6];
        mem_rgs6(uu, uu6, pos);
        
        //sum,diag
        float s = 0e0f;
        float d = 0e0f;
        
        //arms
        for(int i=0; i<6; i++)
        {
//            float g1 = (gg6[i].x>0.0f); //invert
            d += 1.0f;
            s -= uu6[i].x;
        }
        
        
        float4 r;
        r.x = (msh.dx2*b.x - s)/d;
        
        //jacobi
        write_imagef(rr, pos, r);
    }
    return;
}

/*
 ============================
 transfer
 ============================
 */


/*

//project - sum
kernel void ele_prj(const  struct msh_obj   mshc,    //coarse    (out)
                   global float            *rrf,    //fine      (in)
                   global float            *uuc,    //coarse    (out)
                   global float            *bbc)    //coarse    (out)
{
   int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
   int   ele_idx  = utl_idx(ele_pos, mshc.ne);
   
   
   //fine
   int3 pos = 2*ele_pos;
   int3 dim = 2*mshc.ne;
   
   //sum
   float s = 0e0f;
   
   //sum fine
   for(int i=0; i<8; i++)
   {
       int3 adj_pos = pos + off_ele[i];
       int  adj_idx = utl_idx1(adj_pos, dim);
       s += rrf[adj_idx];
   }
   
   //store/reset
   uuc[ele_idx] = 0e0f;
   bbc[ele_idx] = s;
   
   return;
}


//interp - inject
kernel void ele_itp(const  struct msh_obj   mshf,    //fine      (out)
                   global float            *uuc,    //coarse    (in)
                   global float            *uuf)    //fine      (out)
{
   int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
   int   ele_idx  = utl_idx1(ele_pos, mshf.ne);   //fine
   
   //    printf("%2d %v3hlu\n", ele_idx, ele_pos/2);
   
   //coarse
   int3 pos = ele_pos/2;
   int3 dim = mshf.ne/2;
   
   //write - scale
   uuf[ele_idx] += 0.125f*uuc[utl_idx1(pos, dim)];
   
   return;
}


*/
