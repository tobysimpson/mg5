//
//  io.h
//  mg1
//
//  Created by toby on 29.05.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef io_h
#define io_h


#include <stdio.h>
#include "ocl.h"
#include "msh.h"


#define ROOT_WRITE  "/Users/toby/Downloads/"

void wrt_xmf(struct ocl_obj *ocl, struct msh_obj *msh, int l, int f);

void wrt_flt1(struct ocl_obj *ocl, struct msh_obj *msh, cl_mem *buf, char *dsc, int idx, cl_int n_tot);
void wrt_flt3(struct ocl_obj *ocl, struct msh_obj *msh, cl_mem *buf, char *dsc, int idx, cl_int n_tot);
void wrt_flt4(struct ocl_obj *ocl, struct msh_obj *msh, cl_mem *buf, char *dsc, int idx, cl_int n_tot);

void wrt_img1(struct ocl_obj *ocl, cl_mem img, cl_image_desc *dsc, char* tag, int l, int f);

#endif /* io_h */
