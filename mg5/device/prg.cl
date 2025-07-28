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

//read global scalar 2x2x2
void mem_rgs2(read_only image3d_t uu, float4 uu2[8], int4 pos)
{
    uu2[0] = read_imagef(uu, pos + (int4){0,0,0,0});
    uu2[1] = read_imagef(uu, pos + (int4){1,0,0,0});
    uu2[2] = read_imagef(uu, pos + (int4){0,1,0,0});
    uu2[3] = read_imagef(uu, pos + (int4){1,1,0,0});
    uu2[4] = read_imagef(uu, pos + (int4){0,0,1,0});
    uu2[5] = read_imagef(uu, pos + (int4){1,0,1,0});
    uu2[6] = read_imagef(uu, pos + (int4){0,1,1,0});
    uu2[7] = read_imagef(uu, pos + (int4){1,1,1,0});

    return;
}

//read global scalar 6-point stencil
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
    int4 dim = get_image_dim(gg);
    
//   float4 x = msh.dx*(convert_float4(pos - dim/2) + 0.5f);
    
    float4 u = 0e0f + (pos.x==0) - (pos.x==(dim.x-1));  //init
    
    write_imagef(gg, pos, pos.x);
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
    int4 dim = get_image_dim(gg);
    
    float4 x = msh.dx*(convert_float4(pos - dim/2) + 0.5f);
    
    float g1 = sdf_sph(x,(float4){+0.25f, +0.25f, -0.25f, 0.0f}, 0.5f);
    float g2 = sdf_cub(x,(float4){-0.25f, -0.25f, +0.25f, 0.0f}, (float4){0.5f, 0.5f, 0.5f, 0.0f});
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
            float g1 = (gg6[i].x>0.0f); //invert
            d += g1*1.0f;
            s -= g1*uu6[i].x;
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




//project (sum)
kernel void ele_prj(read_only     image3d_t     rr,    //fine      (in)
                    write_only    image3d_t     uu,    //coarse    (out)
                    write_only    image3d_t     bb)    //coarse    (out)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    //fine
    int4 posf = 2*pos;

    //read
    float4 rr2[8];
    mem_rgs2(rr, rr2, posf);
    
    //sum
    float4 s = 0e0f;

    //sum fine
    for(int i=0; i<8; i++)
    {
        s += rr2[i];
    }

    //store/reset
    write_imagef(uu, pos, 0e0f);
    write_imagef(bb, pos, s);    //avg?

    return;
}




//interp (inject)
kernel void ele_itp(read_only   image3d_t   uuc,    //coarse    (in)
                    read_only   image3d_t   rrf,    //fine      (in) holds uuf
                    write_only  image3d_t   uuf)    //fine      (out)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    //read
    float4 uf = read_imagef(rrf, pos);
    float4 uc = read_imagef(uuc, pos/2);

    //write
    write_imagef(uuf, pos, uf + 0.125f*uc);  //avg

    return;
}




