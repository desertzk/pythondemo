cimport dlfcn
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy

DLL = b"libadd.so"
cdef void* _lib_handle = NULL
cdef extern int add(int a, int b)

cdef int __init() except 1:
    global DLL, _lib_handle
    cdef char* dll_path = <char*>malloc(len(DLL) + 1)
    if dll_path == NULL:
        raise MemoryError("Failed to allocate memory")
    strcpy(dll_path, DLL)
    cdef char* error
    if _lib_handle == NULL:
        _lib_handle = dlfcn.dlopen(dll_path, dlfcn.RTLD_NOW)
        if _lib_handle == NULL:
            error = dlfcn.dlerror()
            raise RuntimeError(f"Failed to load {DLL.decode('utf-8')}: {error.decode('utf-8') if error else 'Unknown error'}")
    free(dll_path)  # Clean up
    return 0

def pyadd(int a, int b):
    if __init() != 0:
        raise RuntimeError("Failed to load library")
    cdef void* func_ptr
    with nogil:
        func_ptr = dlfcn.dlsym(_lib_handle, "add")
    if func_ptr == NULL:
        error = dlfcn.dlerror()
        raise RuntimeError(f"Failed to find symbol 'add': {error.decode('utf-8') if error else 'Unknown error'}")
    return (<int (*)(int, int)>func_ptr)(a, b)
