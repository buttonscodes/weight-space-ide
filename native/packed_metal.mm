// ggml 0.29.0 Metal packed matmul; immutable mmap shared with Metal, no weight copy.
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml-metal.h>
#include <dlfcn.h>
#include <cstdint>
#import <Metal/Metal.h>

struct engine { ggml_backend_t backend; ggml_backend_buffer_t weights; uintptr_t base; size_t size; };
extern "C" int wsp_metal_abi() { return 1; }
extern "C" void * wsp_metal_open(const char * library, void * data, size_t size, size_t max_tensor) {
    void * lib = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!lib || !data || !size) return nullptr;
    auto init = (decltype(&ggml_backend_metal_init)) dlsym(lib,"ggml_backend_metal_init");
    if (!init) return nullptr;
    auto backend = init();
    if (!backend) return nullptr;
    auto weights = ggml_backend_dev_buffer_from_host_ptr(ggml_backend_get_device(backend),data,size,max_tensor);
    if (!weights) { ggml_backend_free(backend); return nullptr; }
    ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return new engine{backend,weights,(uintptr_t)data,size};
}
extern "C" void wsp_metal_close(void * handle) {
    if (!handle) return;
    auto e = (engine *)handle;
    ggml_backend_synchronize(e->backend);
    ggml_backend_buffer_free(e->weights);
    ggml_backend_free(e->backend);
    delete e;
}
extern "C" int wsp_metal_memory(uint64_t * allocated, uint64_t * recommended) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -1;
        *allocated = device.currentAllocatedSize;
        *recommended = device.recommendedMaxWorkingSetSize;
        [device release];
        return 0;
    }
}
extern "C" int wsp_metal_matmul(void * handle, int type, const void * packed,
    int64_t rows, int64_t cols, const float * x, int64_t tokens, float * out) {
    auto e = (engine *)handle;
    if (!e || !packed || !x || !out || rows<1 || cols<1 || tokens<1 || tokens>512) return -10;
    switch(type) { case 0: case 1: case 2: case 3: case 6: case 7: case 8:
        case 10: case 11: case 12: case 13: case 14: case 30: break; default: return -10; }
    if (cols % ggml_blck_size((ggml_type)type)) return -10;
    size_t bytes = ggml_row_size((ggml_type)type,cols)*rows;
    if ((uintptr_t)packed < e->base || bytes > e->size || (uintptr_t)packed-e->base > e->size-bytes) return -10;
    if ((uint64_t)(rows+cols)*tokens*4 > 128ULL*1024*1024) return -12;
    ggml_context * ctx = ggml_init({ggml_tensor_overhead()*4 + ggml_graph_overhead_custom(8,false),nullptr,true});
    if (!ctx) return -11;
    auto w = ggml_new_tensor_2d(ctx,(ggml_type)type,cols,rows);
    auto input = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,tokens);
    if (ggml_backend_tensor_alloc(e->weights,w,const_cast<void *>(packed)) != GGML_STATUS_SUCCESS) { ggml_free(ctx); return -13; }
    auto y = ggml_mul_mat(ctx,w,input);
    if (!ggml_backend_supports_op(e->backend,y)) { ggml_free(ctx); return -14; }
    auto activations = ggml_backend_alloc_ctx_tensors(ctx,e->backend); // w already mapped: skipped
    if (!activations) { ggml_free(ctx); return -11; }
    if (ggml_backend_buffer_get_size(activations)>128ULL*1024*1024) {
        ggml_backend_buffer_free(activations); ggml_free(ctx); return -12;
    }
    ggml_backend_tensor_set(input,x,0,ggml_nbytes(input));
    auto graph = ggml_new_graph_custom(ctx,8,false);
    ggml_build_forward_expand(graph,y);
    int status = ggml_backend_graph_compute(e->backend,graph);
    if (status == GGML_STATUS_SUCCESS) ggml_backend_tensor_get(y,out,0,ggml_nbytes(y));
    ggml_backend_synchronize(e->backend);
    ggml_backend_buffer_free(activations); ggml_free(ctx);
    return status;
}
