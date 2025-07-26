//
//  io.c
//  fsi2
//
//  Created by Toby Simpson on 14.04.2025.
//

#include "io.h"



//write xdmf
void wrt_xmf(struct ocl_obj *ocl, struct msh_obj *msh, int l, int f)
{
    FILE* file1;
    char file1_name[250];
//    float* ptr1;
    
    //file name
    sprintf(file1_name, "%s/xmf/msh.%02d.%03u.xmf", ROOT_WRITE, l, f);
    
    //open
    file1 = fopen(file1_name,"w");
    
    fprintf(file1,"<Xdmf>\n");
    fprintf(file1,"  <Domain>\n");
    fprintf(file1,"    <Topology name=\"topo\" TopologyType=\"3DCoRectMesh\" Dimensions=\"%u %u %u\"></Topology>\n", msh->nv.z, msh->nv.y, msh->nv.x);
    fprintf(file1,"      <Geometry name=\"geo\" Type=\"ORIGIN_DXDYDZ\">\n");
    fprintf(file1,"        <!-- Origin -->\n");
//    fprintf(file1,"        <DataItem Format=\"XML\" Dimensions=\"3\">%f %f %f</DataItem>\n", 0e0f, 0e0f, 0e0f);
    fprintf(file1,"        <DataItem Format=\"XML\" Dimensions=\"3\">%f %f %f</DataItem>\n", -msh->dx*(msh->ne.x/2), -msh->dx*(msh->ne.y/2), -msh->dx*(msh->ne.z/2));  //origin at centre
    fprintf(file1,"        <!-- DxDyDz -->\n");
    fprintf(file1,"        <DataItem Format=\"XML\" Dimensions=\"3\">%f %f %f</DataItem>\n", msh->dx, msh->dx, msh->dx);
    fprintf(file1,"      </Geometry>\n");
    fprintf(file1,"      <Grid Name=\"T1\" GridType=\"Uniform\">\n");
    fprintf(file1,"        <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>\n");
    fprintf(file1,"        <Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>\n");
//    fprintf(file1,"         <Attribute Name=\"gg\" Center=\"Cell\" AttributeType=\"Matrix\">\n");
//    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%u %u %u 4\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", msh->ne.z, msh->ne.y, msh->ne.x);
//    fprintf(file1,"             /Users/toby/Downloads/raw/gg.%02u%02u%02u.%03d.raw\n", msh->le.x, msh->le.y, msh->le.z, idx);
//    fprintf(file1,"           </DataItem>\n");
//    fprintf(file1,"         </Attribute>\n");
    fprintf(file1,"         <Attribute Name=\"uu\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%u %u %u 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", msh->nv.z, msh->nv.y, msh->nv.x);
    fprintf(file1,"             /Users/toby/Downloads/raw/uu.%02d.%03d.raw\n", l, f);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
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
void wrt_img1(struct ocl_obj *ocl, cl_mem img, cl_image_desc *dsc, char* tag, int l, int f)
{
    FILE* file1;
    char file1_name[128];
    cl_float* ptr;
    size_t ogn[3] = {0,0,0};
    size_t reg[3] = {dsc->image_width, dsc->image_height, dsc->image_depth};
    size_t rp = sizeof(cl_float)*dsc->image_width;
    size_t sp = sizeof(cl_float)*dsc->image_width*dsc->image_height;
    size_t n = dsc->image_width*dsc->image_height*dsc->image_depth;
    
    //buffer
    sprintf(file1_name, "%s/raw/%s.%02d.%03d.raw", ROOT_WRITE, tag, l, f);
    file1 = fopen(file1_name,"wb");
    ptr = clEnqueueMapImage(ocl->command_queue, img, CL_TRUE, CL_MAP_READ, ogn, reg, &rp, &sp, 0, NULL, NULL, NULL);
    fwrite(ptr, sizeof(cl_float), n, file1);
    clEnqueueUnmapMemObject(ocl->command_queue, img, ptr, 0, NULL, NULL);
    
    //clean up
    fclose(file1);
    
    return;
}
