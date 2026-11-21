// Narrow C ABI over ggml 0.29.0 CPU packed matmul. No dense weight allocation.
#include <ggml.h>
#include <ggml-cpu.h>
#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>

static decltype(&ggml_get_type_traits_cpu) cpu_traits;
static decltype(&ggml_graph_plan) cpu_plan;
static decltype(&ggml_graph_compute) cpu_compute;
extern "C" int wsp_init(const char * path) {
    void * lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) return -1;
    cpu_traits = (decltype(cpu_traits))dlsym(lib, "ggml_get_type_traits_cpu");
    cpu_plan = (decltype(cpu_plan))dlsym(lib, "ggml_graph_plan");
    cpu_compute = (decltype(cpu_compute))dlsym(lib, "ggml_graph_compute");
    return cpu_traits && cpu_plan && cpu_compute ? 0 : -2;
}

extern "C" int wsp_abi() { return 1; }
extern "C" int wsp_supported(int type) {
    // Deliberate allowlist, checked before ggml's assertion-based API.
    switch (type) { case 0: case 1: case 2: case 3: case 6: case 7:
                   case 8: case 10: case 11: case 12: case 13: case 14: case 30: break;
                   default: return 0; }
    return cpu_traits && cpu_traits((ggml_type)type)->vec_dot != nullptr;
}
extern "C" int wsp_matmul(int type, const void * packed, int64_t rows, int64_t cols,
                           const float * x, int64_t tokens, float * out, int threads) {
    if (!wsp_supported(type) || !packed || !x || !out || rows < 1 || cols < 1 ||
        tokens < 1 || tokens > 2048 || threads < 1 || threads > 32 ||
        cols % ggml_blck_size((ggml_type)type)) return -10;
    ggml_context * ctx = ggml_init({ggml_tensor_overhead()*4 + ggml_graph_overhead_custom(8, false), nullptr, true});
    if (!ctx) return -11;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, (ggml_type)type, cols, rows);
    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, tokens);
    w->data = const_cast<void *>(packed); // read-only mmap; ggml_mul_mat never writes its inputs
    in->data = const_cast<float *>(x);
    ggml_tensor * y = ggml_mul_mat(ctx, w, in);
    y->data = out;
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, y);
    ggml_cplan plan = cpu_plan(graph, threads, nullptr);
    // Only activation packing / per-thread scratch. Never a dense weight matrix.
    if (plan.work_size > 128ULL*1024*1024) { ggml_free(ctx); return -12; }
    plan.work_data = (uint8_t *)malloc(plan.work_size ? plan.work_size : 1);
    if (!plan.work_data) { ggml_free(ctx); return -11; }
    int status = cpu_compute(graph, &plan);
    free(plan.work_data);
    ggml_free(ctx);
    return status;
}
