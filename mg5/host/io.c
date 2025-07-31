//
//  io.c
//  fsi2
//
//  Created by Toby Simpson on 14.04.2025.
//

#include "io.h"


//write xdmf
void wrt_xmf(struct ocl_obj *ocl, struct lvl_obj *lvl, int l, int f)
{
    FILE* file1;
    char file1_name[250];
    
    //file name
    sprintf(file1_name, "%s/xmf/msh.%02d.%03u.xmf", ROOT_WRITE, l, f);
    
    //open
    file1 = fopen(file1_name,"w");
    
    fprintf(file1,"<Xdmf>\n");
    fprintf(file1,"  <Domain>\n");
    fprintf(file1,"    <Topology name=\"topo\" TopologyType=\"3DCoRectMesh\" Dimensions=\"%zu %zu %zu\"></Topology>\n", lvl->vxl.n[2], lvl->vxl.n[1], lvl->vxl.n[0]);
    fprintf(file1,"      <Geometry name=\"geo\" Type=\"ORIGIN_DXDYDZ\">\n");
    fprintf(file1,"        <!-- Origin -->\n");
//    fprintf(file1,"        <DataItem Format=\"XML\" Dimensions=\"3\">%f %f %f</DataItem>\n", 0e0f, 0e0f, 0e0f);
    fprintf(file1,"        <DataItem Format=\"XML\" Dimensions=\"3\">%f %f %f</DataItem>\n", -lvl->msh.dx*(lvl->ele.n[0]/2), -lvl->msh.dx*(lvl->ele.n[1]/2), -lvl->msh.dx*(lvl->ele.n[2]/2));  //origin at centre
    fprintf(file1,"        <!-- DxDyDz -->\n");
    fprintf(file1,"        <DataItem Format=\"XML\" Dimensions=\"3\">%f %f %f</DataItem>\n", lvl->msh.dx, lvl->msh.dx, lvl->msh.dx);
    fprintf(file1,"      </Geometry>\n");
    fprintf(file1,"      <Grid Name=\"T1\" GridType=\"Uniform\">\n");
    fprintf(file1,"        <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>\n");
    fprintf(file1,"        <Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>\n");

    fprintf(file1,"         <Attribute Name=\"gg\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%zu %zu %zu 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", lvl->vxl.n[2],lvl->vxl.n[1],lvl->vxl.n[0]);
    fprintf(file1,"             /Users/toby/Downloads/raw/gg.%02d.%03d.raw\n", l, f);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    
    fprintf(file1,"         <Attribute Name=\"uu\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%zu %zu %zu 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", lvl->vxl.n[2],lvl->vxl.n[1],lvl->vxl.n[0]);
    fprintf(file1,"             /Users/toby/Downloads/raw/uu.%02d.%03d.raw\n", l, f);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    
    fprintf(file1,"         <Attribute Name=\"bb\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%zu %zu %zu 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", lvl->vxl.n[2],lvl->vxl.n[1],lvl->vxl.n[0]);
    fprintf(file1,"             /Users/toby/Downloads/raw/bb.%02d.%03d.raw\n", l, f);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    
    fprintf(file1,"         <Attribute Name=\"rr\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%zu %zu %zu 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", lvl->vxl.n[2],lvl->vxl.n[1],lvl->vxl.n[0]);
    fprintf(file1,"             /Users/toby/Downloads/raw/rr.%02d.%03d.raw\n", l, f);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    
//    fprintf(file1,"         <Attribute Name=\"gg\" Center=\"Cell\" AttributeType=\"Matrix\">\n");
//    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%u %u %u 4\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", msh->ne.z, msh->ne.y, msh->ne.x);
//    fprintf(file1,"             /Users/toby/Downloads/raw/gg.%02u%02u%02u.%03d.raw\n", msh->le.x, msh->le.y, msh->le.z, idx);
//    fprintf(file1,"           </DataItem>\n");
//    fprintf(file1,"         </Attribute>\n");
//    fprintf(file1,"         <Attribute Name=\"vv\" Center=\"Node\" AttributeType=\"Vector\">\n");
//    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%u %u %u 3\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", msh->nv.z, msh->nv.y, msh->nv.x);
//    fprintf(file1,"             /Users/toby/Downloads/raw/vv.%02u%02u%02u.%03d.raw\n", msh->le.x, msh->le.y, msh->le.z, idx);
//    fprintf(file1,"           </DataItem>\n");
//    fprintf(file1,"         </Attribute>\n");
    fprintf(file1,"    </Grid>\n");
    fprintf(file1," </Domain>\n");
    fprintf(file1,"</Xdmf>\n");
    
    //clean up
    fclose(file1);
}


//float
void wrt_img(struct ocl_obj *ocl, cl_mem img, char* tag, int l, int f)
{
    FILE* file1;
    char file1_name[128];
    cl_float* ptr;
    
    size_t ogn[3] = {0,0,0};
    size_t ele_sz;
    size_t dim[3];
    size_t rp;
    size_t sp;
    
    ocl->err = clGetImageInfo(img, CL_IMAGE_ELEMENT_SIZE,   sizeof(size_t), &ele_sz,    NULL);
    ocl->err = clGetImageInfo(img, CL_IMAGE_WIDTH,          sizeof(size_t), &dim[0],      NULL);
    ocl->err = clGetImageInfo(img, CL_IMAGE_HEIGHT,         sizeof(size_t), &dim[1],      NULL);
    ocl->err = clGetImageInfo(img, CL_IMAGE_DEPTH,          sizeof(size_t), &dim[2],      NULL);
    ocl->err = clGetImageInfo(img, CL_IMAGE_ROW_PITCH,      sizeof(size_t), &rp,        NULL);
    ocl->err = clGetImageInfo(img, CL_IMAGE_SLICE_PITCH,    sizeof(size_t), &sp,        NULL);
    
    //buffer
    sprintf(file1_name, "%s/raw/%s.%02d.%03d.raw", ROOT_WRITE, tag, l, f);
    file1 = fopen(file1_name,"wb");
    ptr = clEnqueueMapImage(ocl->command_queue, img, CL_TRUE, CL_MAP_READ, ogn, dim, &rp, &sp, 0, NULL, NULL, NULL);
    fwrite(ptr, ele_sz, dim[0]*dim[1]*dim[2], file1);
    clEnqueueUnmapMemObject(ocl->command_queue, img, ptr, 0, NULL, NULL);
    
    //clean up
    fclose(file1);
    
    return;
}
