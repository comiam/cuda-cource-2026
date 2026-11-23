#pragma once

__global__ void addCoalesced(const float *d_x, const float *d_y, float *d_out, int n);
__global__ void addNonCoalesced(const float *d_x, const float *d_y, float *d_out, int n);

__global__ void saxpyCoalesced(const float *d_x, const float *d_y, float *d_out, float alpha,
                               int n);
__global__ void saxpyNonCoalesced(const float *d_x, const float *d_y, float *d_out, float alpha,
                                  int n);

__global__ void reluCoalesced(const float *d_x, float *d_out, int n);
__global__ void reluNonCoalesced(const float *d_x, float *d_out, int n);
