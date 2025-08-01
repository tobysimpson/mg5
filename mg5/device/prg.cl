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


constant int4 off_fac[6] = {{-1,0,0,0},{+1,0,0,0},{0,-1,0,0},{0,+1,0,0},{0,0,-1,0},{0,0,+1,0}};

//global index
int utl_idx(int4 pos, int4 dim)
{
    return pos.x + dim.x*pos.y + dim.x*dim.y*pos.z;
}

//in-bounds
int utl_bnd(int4 pos, int4 dim)
{
    return all(pos.xyz>-1)&&all(pos.xyz<dim.xyz);
}

//on the border
int utl_bdr(int4 pos, int4 dim)
{
    return any(pos.xyz==0)||any(pos.xyz==(dim.xyz-1));
}


/*
 ============================
 stencil
 ============================
 */

//stencil 6-point laplacian zero neumann wrt domain
void stl_lap(read_only image3d_t uu, int4 pos, int4 dim, float4 *s, float *d)
{
    for(int i=0; i<6; i++)
    {
        int4 adj = pos + off_fac[i];
        int  bnd = utl_bnd(adj, dim);
        
        *d -= bnd;
        *s += bnd*read_imagef(uu, adj);
    }

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


float sdf_g1(float4 x)
{
    float g1 = sdf_sph(x,(float4){+0.25f, +0.25f, -0.25f, 0.0f}, 0.5f);
    float g2 = sdf_cub(x,(float4){-0.25f, -0.25f, +0.25f, 0.0f}, (float4){0.5f, 0.5f, 0.5f, 0.0f});
    float g = min(g1,g2);
    
     return g;
}

/*
 ============================
 ini
 ============================
 */


//ini
kernel void vxl_ini(const struct msh_obj  msh,
                    write_only image3d_t  gg,
                    write_only image3d_t  bb,
                    write_only image3d_t  uu,
                    write_only image3d_t  rr)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    int4 dim = get_image_dim(gg);
    
//   float4 x = msh.dx*(convert_float4(pos - (dim-1)/2)); //origin at centre
    float4 u = 0e0f + (pos.x==0) - (pos.x==(dim.x-1));  //init
    
    //dirichlet
    float g = utl_bdr(pos, dim);
    
    write_imagef(gg, pos, g);
    write_imagef(uu, pos, u);
    write_imagef(bb, pos, 0.0f);
    write_imagef(rr, pos, 0.0f);

    return;
}

/*
 ============================
 solve
 ============================
 */

//jacobi
kernel void vxl_jac(const struct msh_obj  msh,
                    read_only     image3d_t       gg,
                    read_only     image3d_t       bb,
                    read_only     image3d_t       uu,
                    write_only    image3d_t       rr)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    int4 dim = get_image_dim(gg);
    
    //read
    float4 g = read_imagef(gg, pos);
    float4 b = read_imagef(bb, pos);
    float4 u = read_imagef(uu, pos);

    //geom
    if(g.x<=0.0f)
    {
        //sum,diag
        float4 s = 0e0f;
        float  d = 0e0f;
        
        //stencil
        stl_lap(uu, pos, dim, &s, &d);

        //update, damp
        u.x += 0.89f*(msh.dx2*b.x - (s.x + d*u.x))/d;
    }
    
    //write
    write_imagef(rr, pos, u);
    
    return;
}


//residual
kernel void vxl_res(const struct msh_obj  msh,
                    read_only     image3d_t       gg,
                    read_only     image3d_t       bb,
                    read_only     image3d_t       uu,
                    write_only    image3d_t       rr)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    int4 dim = get_image_dim(gg);
    
    //read
    float4 g = read_imagef(gg, pos);
    float4 b = read_imagef(bb, pos);
    float4 u = read_imagef(uu, pos);
    float4 r = 0.0f;
    
    //geom
    if(g.x<=0.0f)
    {
        //sum,diag
        float4 s = 0e0f;
        float  d = 0e0f;
        
        //stencil
        stl_lap(uu, pos, dim, &s, &d);
        
        //res
        r = b.x - msh.rdx2*(s + d*u.x);
    }
    
    //write
    write_imagef(rr, pos, r);
    
    return;
}

//forward
kernel void vxl_fwd(const struct msh_obj  msh,
                    read_only   image3d_t       gg,
                    write_only  image3d_t       bb,
                    read_only   image3d_t       uu)
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    int4 dim = get_image_dim(gg);
    
    //read
    float4 g = read_imagef(gg, pos);
    float4 b = 0.0f;
    float4 u = read_imagef(uu, pos);
    
    //sum,diag
    float4 s = 0e0f;
    float  d = 0e0f;
    
    //geom
    if(g.x<=0.0f)
    {
        //stencil
        stl_lap(uu, pos, dim, &s, &d);
        
        b = msh.rdx2*(s + d*u.x);
    }
    
    //write
    write_imagef(bb, pos, d);
    
    return;
}

/*
 ============================
 transfer
 ============================
 */


//project
kernel void vxl_prj(read_only     image3d_t     rr,    //fine
                    write_only    image3d_t     bb)    //coarse
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    //fine
    int4 posf = 2*pos;
    
    //read
    float4 r = read_imagef(rr, posf);
    
    //write
    write_imagef(bb, pos, r);

    return;
}



//interpolate
kernel void vxl_itp(read_only   image3d_t   uuc,    //coarse    (in)
                    write_only  image3d_t   uuf,    //fine      (out)
                    read_only   image3d_t   rrf)    //fine      (in) holds uuf
{
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    
    //coarse
    float4 posc = 0.5f*convert_float4(pos);
    
    //round
    int4 pos0 = convert_int4(floor(posc));
    int4 pos1 = convert_int4(ceil(posc));
    
    //sum
    float4 s = 0.0f;
    s += read_imagef(uuc, (int4){pos0.x, pos0.y, pos0.z, 0});
    s += read_imagef(uuc, (int4){pos1.x, pos0.y, pos0.z, 0});
    s += read_imagef(uuc, (int4){pos0.x, pos1.y, pos0.z, 0});
    s += read_imagef(uuc, (int4){pos1.x, pos1.y, pos0.z, 0});
    s += read_imagef(uuc, (int4){pos0.x, pos0.y, pos1.z, 0});
    s += read_imagef(uuc, (int4){pos1.x, pos0.y, pos1.z, 0});
    s += read_imagef(uuc, (int4){pos0.x, pos1.y, pos1.z, 0});
    s += read_imagef(uuc, (int4){pos1.x, pos1.y, pos1.z, 0});
    
    //read
    float4 r = read_imagef(rrf, pos);

    //write
    write_imagef(uuf, pos, r + 0.125f*s);  //avg

    return;
}



//fold
kernel void vec_sum(global float *uu,
                    const  int   n)
{
    int i = get_global_id(0);
    int m = get_global_size(0);

//    printf("%d %d %d %f %f\n",i, n, m, uu[i], uu[m+i]);
    
    if((m+i)<n)
    {
        uu[i] += uu[m+i];
    }
      
    return;
}
