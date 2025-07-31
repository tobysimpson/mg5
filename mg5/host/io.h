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
#include "mg.h"


#define ROOT_WRITE  "/Users/toby/Downloads/"

void wrt_xmf(struct ocl_obj *ocl, struct lvl_obj *lvl, int l, int f);
void wrt_img(struct ocl_obj *ocl, cl_mem img, char* tag, int l, int f);

#endif /* io_h */
