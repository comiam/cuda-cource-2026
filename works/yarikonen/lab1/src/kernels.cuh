#pragma once

#include <cstddef>

__global__ void add(const float *dX, const float *dY, float *dAdd, std::size_t n,
                    bool noncoalesced);
__global__ void saxpy(const float *dX, const float *dY, float *dSaxpy, float alpha, std::size_t n,
                      bool noncoalesced);
__global__ void relu(const float *dX, float *dRelu, std::size_t n, bool noncoalesced);
