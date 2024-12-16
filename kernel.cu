#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

void 
Acceleration_CPU(
    float* X, 
    float* Y, 
    float* AX, 
    float* AY, 
    int nt, 
    int N, 
    int id
) {
    float ax = 0.f, ay = 0.f, xx, yy, rr;
    int sh = (nt - 1) * N;

    for (int j = 0; j < N; j++) {
        if (j != id) {
            xx = X[j + sh] - X[id + sh];
            yy = Y[j + sh] - Y[id + sh];
            rr = sqrtf(xx * xx + yy * yy);
            if (rr > 0.01f) {
                rr = 10.f / (rr * rr * rr);
                ax += xx * rr;
                ay += yy * rr;
            }
        }
    }
    AX[id] = ax;
    AY[id] = ay;
}

void 
Position_CPU(
    float* X, 
    float* Y, 
    float* VX, 
    float* VY, 
    float* AX, 
    float* AY, 
    float tau, 
    int nt, 
    int N, 
    int id
) {
    int sh = (nt - 1) * N;
    X[id + nt * N] = X[id + sh] + VX[id] * tau + AX[id] * tau * tau * 0.5f;
    Y[id + nt * N] = Y[id + sh] + VY[id] * tau + AY[id] * tau * tau * 0.5f;
    VX[id] += AX[id] * tau;
    VY[id] += AY[id] * tau;
}

__global__ void 
Position_GPU(
    float* X, 
    float* Y, 
    float* VX, 
    float* VY, 
    float* AX, 
    float* AY, 
    float tau, 
    int nt, 
    int N
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int sh = (nt - 1) * N;

    X[id + nt * N] = X[id + sh] + VX[id] * tau + AX[id] * tau * tau * 0.5f;
    Y[id + nt * N] = Y[id + sh] + VY[id] * tau + AY[id] * tau * tau * 0.5f;
    VX[id] += AX[id] * tau;
    VY[id] += AY[id] * tau;
}

__global__ void 
Acceleration_GPU(
    float* X, 
    float* Y, 
    float* AX, 
    float* AY, 
    int nt, 
    int N
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float ax = 0.f, ay = 0.f, xx, yy, rr;
    int sh = (nt - 1) * N;

    for (int j = 0; j < N; j++) {
        if (j != id) {
            xx = X[j + sh] - X[id + sh];
            yy = Y[j + sh] - Y[id + sh];
            rr = sqrtf(xx * xx + yy * yy);
            if (rr > 0.01f) {
                rr = 10.f / (rr * rr * rr);
                ax += xx * rr;
                ay += yy * rr;
            }
        }
    }
    AX[id] = ax;
    AY[id] = ay;
}

__global__ void 
Acceleration_Shared(
    float* X, 
    float* Y, 
    float* AX, 
    float* AY,
    int nt, 
    int N, 
    int N_block
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float ax = 0.f;
    float ay = 0.f;
    float xx, yy, rr;
    int sh = (nt - 1) * N;
    float xxx = X[id + sh];
    float yyy = Y[id + sh];
    __shared__ float Xs[256];
    __shared__ float Ys[256];

    for (int i = 0; i < N_block; i++) {
        Xs[threadIdx.x] = X[threadIdx.x + i * blockDim.x + sh];
        Ys[threadIdx.x] = Y[threadIdx.x + i * blockDim.x + sh];
        __syncthreads();

        for (int j = 0; j < blockDim.x; j++) {
            if ((j + i * blockDim.x) != id) {
                xx = Xs[j] - xxx;
                yy = Ys[j] - yyy;
                rr = sqrtf(xx * xx + yy * yy);
                if (rr > 0.01f) {
                    rr = 10.f / (rr * rr * rr);
                    ax += xx * rr;
                    ay += yy * rr;
                }
            }
        }
        __syncthreads();
    }
    AX[id] = ax;
    AY[id] = ay;
}


int main() {

    int NN[] = { 10240, 20480 };
    int NT = 10;
    float tau = 0.001f;

    float timerGpuGlobal, timerGpuShared, timerCpu;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 2; ++i) {
      
        int N = NN[i];
        
        float* hX, * hY, * hVX, * hVY, * hAX, * hAY;
        unsigned int mem_size = sizeof(float) * N;
        unsigned int mem_size_big = sizeof(float) * NT * N;

        hX = (float*)malloc(mem_size_big);
        hY = (float*)malloc(mem_size_big);
        hVX = (float*)malloc(mem_size);
        hVY = (float*)malloc(mem_size);
        hAX = (float*)malloc(mem_size);
        hAY = (float*)malloc(mem_size);

        for (int j = 0; j < N; j++) {
            float phi = (float)rand();
            hX[j] = rand() * cosf(phi) * 1.e-4f;
            hY[j] = rand() * sinf(phi) * 1.e-4f;
            float vv = (hX[j] * hX[j] + hY[j] * hY[j]) * 10.f;
            hVX[j] = -vv * sinf(phi);
            hVY[j] = vv * cosf(phi);
        }

        // for global
        float* dX, * dY, * dVX, * dVY, * dAX, * dAY;
        cudaMalloc((void**)&dX, mem_size_big);
        cudaMalloc((void**)&dY, mem_size_big);
        cudaMalloc((void**)&dVX, mem_size);
        cudaMalloc((void**)&dVY, mem_size);
        cudaMalloc((void**)&dAX, mem_size);
        cudaMalloc((void**)&dAY, mem_size);

        // for shared
        float* dXS, * dYS, * dVXS, * dVYS, * dAXS, * dAYS;
        cudaMalloc((void**)&dXS, mem_size_big);
        cudaMalloc((void**)&dYS, mem_size_big);
        cudaMalloc((void**)&dVXS, mem_size);
        cudaMalloc((void**)&dVYS, mem_size);
        cudaMalloc((void**)&dAXS, mem_size);
        cudaMalloc((void**)&dAYS, mem_size);

        int N_thread = 256;
        int N_block = N / N_thread;

        // ----------- GPU global -----------
        cudaEventRecord(start, 0);
        cudaMemcpy(dX, hX, mem_size_big, cudaMemcpyHostToDevice);
        cudaMemcpy(dY, hY, mem_size_big, cudaMemcpyHostToDevice);
        cudaMemcpy(dVX, hVX, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dVY, hVY, mem_size, cudaMemcpyHostToDevice);

        for (int j = 1; j < NT; j++) {
            Acceleration_Shared << <N_block, N_thread >> > (dX, dY, dAX, dAY, j, N, N_block);
            Position_GPU << <N_block, N_thread >> > (dX, dY, dVX, dVY, dAX, dAY, tau, j, N);
        }

        cudaMemcpy(hX, dX, mem_size_big, cudaMemcpyDeviceToHost);
        cudaMemcpy(hY, dY, mem_size_big, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timerGpuGlobal, start, stop);

        // ----------- GPU shared -----------
        cudaEventRecord(start, 0);
        cudaMemcpy(dXS, hX, mem_size_big, cudaMemcpyHostToDevice);
        cudaMemcpy(dYS, hY, mem_size_big, cudaMemcpyHostToDevice);
        cudaMemcpy(dVXS, hVX, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dVYS, hVY, mem_size, cudaMemcpyHostToDevice);

        for (int j = 1; j < NT; j++) {
            Acceleration_Shared << <N_block, N_thread >> > (dXS, dYS, dAXS, dAYS, j, N, N_block);
            Position_GPU << <N_block, N_thread >> > (dXS, dYS, dVXS, dVYS, dAXS, dAYS, tau, j, N);
        }

        cudaMemcpy(hX, dXS, mem_size_big, cudaMemcpyDeviceToHost);
        cudaMemcpy(hY, dYS, mem_size_big, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timerGpuShared, start, stop);

        // ----------- CPU -----------
        auto startCPU = std::chrono::high_resolution_clock::now();
        for (int j = 1; j < NT; j++) {
            for (int id = 0; id < N; id++) {
                Acceleration_CPU(hX, hY, hAX, hAY, j, N, id);
                Position_CPU(hX, hY, hVX, hVY, hAX, hAY, tau, j, N, id);
            }
        }

        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpuTime = endCPU - startCPU;
        timerCpu = cpuTime.count();


        std::cout << "CPU time: " << cpuTime.count() << " ms\n";
        std::cout << "GPU (Global memory) time: " << timerGpuGlobal << " ms\n";
        std::cout << "GPU (Shared memory) time: " << timerGpuShared << " ms\n";
        std::cout << "Speedup (Global): " << cpuTime.count() / timerGpuGlobal << "x\n";
        std::cout << "Speedup (Shared): " << cpuTime.count() / timerGpuShared << "x\n";

        free(hX); 
        free(hY); 
        free(hVX); 
        free(hVY); 
        free(hAX); 
        free(hAY);
        cudaFree(dX); 
        cudaFree(dY); 
        cudaFree(dVX); 
        cudaFree(dVY);

    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
