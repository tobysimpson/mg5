//
//  msh.c
//  mg5
//
//  Created by Toby Simpson on 26/07/2025.
//

#include "msh.h"

//init
void msh_ini(struct msh_obj *msh)
{
    msh->dx2        = msh->dx*msh->dx;
    msh->rdx2       = 1e0f/msh->dx2;

    return;
}




