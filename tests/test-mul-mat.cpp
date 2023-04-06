//#define GGML_PERF
//#define GGML_DEBUG
#include "ggml.h"
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <ostream>
#include <sstream>
#include <stdio.h>
#include <string>

void formatsizes(const ggml_tensor * t, std::ostream & str) {
    str << "[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "]";
}

void print2d(const ggml_tensor * t) {
    float * fd = (float *) t->data;
    int nb1 = t->ne[0];
    for (int i1 = 0; i1 < t->ne[1]; i1++) {
        for (int i0 = 0; i0 < t->ne[0]; i0++) {
            printf("%f ", fd[i1*nb1 + i0]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {

    //  A      * B        C
    // (K x N) * (K, M) = (N, M)
    const int64_t N = 1024;
    const int64_t K = 4096;
    const int64_t M = 2048;
    const int64_t L = 512;
    const int64_t O = 8192;

    // C * D = ML
    //(N, M) (N, L) = (M, L)

    // ML * MO = LO

    const int64_t MAGIC = 100;

    const size_t BUF_SIZE = MAGIC * (4 * N * M + 4 * K * M + 4 * N * M);

    bool open_cl = argc>1;

    ggml_cl_context * cl_ctx = open_cl ? ggml_init_cl() : NULL;

    ggml_init_params params = {
        /*.mem_size   =*/ BUF_SIZE,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
        /*.cl_ctx     =*/ cl_ctx,
    };

    ggml_time_init();
    ggml_context * ctx = ggml_init(params);

    auto * kn =
        ggml_set_f32(
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N), 0.125);
    ggml_set_name(kn, "kn");
    if (ggml_cl_enabled(ctx)) ggml_tensor_upload_cl(ctx, kn);

    auto * km =
        ggml_set_f32(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M), 0.125);
    ggml_set_name(km, "km");
    if (ggml_cl_enabled(ctx)) ggml_tensor_upload_cl(ctx, km);

    auto * nm = ggml_mul_mat(ctx, kn, km);
    ggml_set_name(nm, "nm");

    auto * nl =
        ggml_set_f32(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, L), 1);
    ggml_set_name(nl, "nl");
    if (ggml_cl_enabled(ctx)) ggml_tensor_upload_cl(ctx, nl);

    auto * ml = ggml_mul_mat(ctx, nm, nl);
    ggml_set_name(ml, "ml");
    if (ggml_cl_enabled(ctx)) ggml_tensor_upload_cl(ctx, ml);

    auto * mo =
        ggml_set_f32(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, O), 1);
    ggml_set_name(mo, "mo");
    if (ggml_cl_enabled(ctx)) ggml_tensor_upload_cl(ctx, mo);

    auto * lo = ggml_mul_mat(ctx, ml, mo);

    auto * lm = ggml_dup(ctx, ggml_transpose(ctx, ml));

    auto * om = ggml_mul_mat(ctx, lo, lm);

    auto * mean =  ggml_mean(ctx, om);

    ggml_tensor * result = ggml_mean(ctx, ggml_transpose(ctx, mean)); // (M, L)

    ggml_cgraph gf = {};
    gf.n_threads = 1;

    ggml_build_forward_expand(&gf, result);
    ggml_graph_compute(ctx, &gf);

    std::stringstream sizes;
    sizes << "\n kn "; formatsizes(kn, sizes);
    sizes << "\n km "; formatsizes(km, sizes);
    sizes << "\n nm "; formatsizes(nm, sizes);
    sizes << "\n ml "; formatsizes(ml, sizes);
    sizes << "\n mo "; formatsizes(mo, sizes);
    sizes << "\n lo "; formatsizes(lo, sizes);
    sizes << "\n lm "; formatsizes(lm, sizes);
    sizes << "\n om "; formatsizes(om, sizes);
    sizes << "\n mean "; formatsizes(mean, sizes);
    sizes << "\n result "; formatsizes(result, sizes);

    printf("sizes \n %s\n", sizes.str().c_str());


    /*printf("data\n");
    printf(" kn\n");print2d(kn);
    printf(" km\n");print2d(km);
    printf(" nm\n");print2d(nm);
    printf(" ml\n");print2d(ml);
    printf(" mo\n");print2d(mo);
    printf(" lo\n");print2d(lo);
    printf(" lm\n");print2d(lm);
    printf(" om\n");print2d(om);*/


    //printf(" mean\n");print2d(mean);
    printf(" result\n");print2d(result);

    ggml_free(ctx);
    ggml_free_cl(cl_ctx);

    return 0;
}
