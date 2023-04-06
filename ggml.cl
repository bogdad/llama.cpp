#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void dup_f32(
    const int Xb0, // nb00
    const int Xb1, // nb01
    const int Zb0, // nb0
    const int Zb1, // nb1
    __global const float *X,
    __global float *Z
    ) {

    const int x0 = get_global_id(0); // Xe0
    const int x1 = get_global_id(1); // Xe1

    const int Xbe0 = Xb0 / sizeof(float);
    const int Xbe1 = Xb1 / sizeof(float);
    const int Zbe0 = Zb0 / sizeof(float);
    const int Zbe1 = Zb1 / sizeof(float);

    int zid = Zbe0 * x0 + Zbe1 * x1;
    int xid = Xbe0 * x0 + Xbe1 * x1;
    Z[zid] = X[xid];
    // printf("(%d,%d) x[%d]=%f z[%d]=%f\n", x0, x1, xid, X[xid], zid, Z[zid]);
}


__kernel void matmul4D_f32(
    const int Ze0, // Xe1
    const int Ze1, // Ye1
    const int Xe0, // == Ye0 == K shared dimension
    __global const float *X,
    __global const float *Y,
    __global float *Z
    ) {

    const int Xe1 = Ze0;
    const int Ye1 = Ze1;
    const int Ye0 = Xe0;

    const int z0 = get_global_id(0); // Ze0
    const int z1 = get_global_id(1); // Ze1

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k = 0; k < Xe0; k++) {
        acc += X[k + Xe0*z0] * Y[k + Ye0*z1];
    }

    // Store the result
    Z[z0 + z1*Ze0] = acc;
    //printf("ccccc z(%d, %d), x(%d, %d), y(%d, %d) Z[%d,%d] = %f (%f,%f)\n",Ze0, Ze1, Xe0, Xe1, Ye0, Ye1, z0, z1, acc,
    //    X[0+Xe1*z0], Y[0+Ye1*z1]);
}

__kernel void matmul4D_f16_f32(
    const int Ze0, // Xe1
    const int Ze1, // Ye1
    const int Xe0, // == Ye0 == K shared dimension
    __global const half *X,
    __global const float *Y,
    __global float *Z
    ) {

    const int Xe1 = Ze0;
    const int Ye1 = Ze1;
    const int Ye0 = Xe0;

    const int z0 = get_global_id(0); // Ze0
    const int z1 = get_global_id(1); // Ze1

    // Compute a single element (loop over K)
    float acc = 0.0f;
    float ys = 0.0f;
    float xs = 0.0f;
    for (int k=0; k < Xe0; k++) {
        //ys += Y[k + Ye1*z1];
        //xs += X[k + Xe1*z0];
        acc += X[k + Xe1*z0] ;//* Y[k + Ye1*z1];
    }

    // Store the result
    Z[z0 + z1*Ze1] = acc;
    printf("ccccc z(%d, %d), x(%d, %d), y(%d, %d) Z[%d + %d * %d] = %f (%f,%f)\n",Ze0, Ze1, Xe0, Xe1, Ye0, Ye1, z0, z1, Ze1, acc, xs, ys);

}
