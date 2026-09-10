// Editor-only prelude for clangd. The engine never compiles this file.
//
// src/kernels.rs prepends the stdlib include and namespace when it
// assembles the fragments, so the fragments carry neither. kernels/.clangd
// force-includes this header so clangd sees the same declarations. The
// metal-lsp stubs also lack a few names the kernels use; they are supplied
// here.
#pragma once

#include <metal_stdlib>

using namespace metal;

using bfloat = __bf16;

template <typename T, int Cols, int Rows = Cols>
simdgroup_matrix<T, Cols, Rows> make_filled_simdgroup_matrix(T value);

#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
