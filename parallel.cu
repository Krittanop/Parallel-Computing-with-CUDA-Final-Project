#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__global__ void horizontal_integral(unsigned char *input, long *output, int width, int height, int C, int channel) {
  extern __shared__ long arr[];

  int y = blockIdx.x;
  if (y >= height) return;

  int tid = threadIdx.x;
  int i = tid;


  // Load input into shared memory
  if (i < width) {
    int idx = (y * width + i) * C + channel;
    arr[tid] = input[idx];
  } else {
    arr[tid] = 0;
  }
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < blockDim.x) {
      arr[index] += arr[index - stride];
    }
    __syncthreads();
  }

  if (tid == 0) arr[blockDim.x - 1] = 0;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < blockDim.x) {
      long t = arr[index - stride];
      arr[index - stride] = arr[index];
      arr[index] += t;
    }
    __syncthreads();
  }

  if (i < width) {
    long value = arr[tid];
    // convert exclusive -> inclusive prefix: add original element
    value += input[(y * width + i) * C + channel];
    output[y * width + i] = value;
  }
}


__global__ void vertical_integral(long *input, long *output, int width, int height) {
  extern __shared__ long arr[];

  int x = blockIdx.x;
  if (x >= width) return;

  int tid = threadIdx.x;
  int y = tid;

  if (y < height) {
    arr[tid] = input[y * width + x];
  } else {
    arr[tid] = 0;
  }
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < blockDim.x) {
      arr[index] += arr[index - stride];
    }
    __syncthreads();
  }

  if (tid == 0) arr[blockDim.x - 1] = 0;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < blockDim.x) {
      long t = arr[index - stride];
      arr[index - stride] = arr[index];
      arr[index] += t;
    }
    __syncthreads();
  }

  if (y < height) {
    long value = arr[tid] + input[y * width + x];
    output[y * width + x] = value;
  }
}


__global__ void blur_from_integral(long **S_device, unsigned char *output, int width, int height, int channel, int r) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= width || y >= height) return;
  
  int x1 = max(0, x - r);
  int y1 = max(0, y - r);
  int x2 = min(width - 1, x + r);
  int y2 = min(height - 1, y + r);
  
  int area = (x2 - x1 + 1) * (y2 - y1 + 1);
  float inv_area = 1.0f / area;
  
  // Pre-compute indices (if all channels use same integral image layout)
  int idx_tl = (y1 - 1) * width + (x1 - 1);  // top-left
  int idx_tr = (y1 - 1) * width + x2;        // top-right
  int idx_bl = y2 * width + (x1 - 1);        // bottom-left
  int idx_br = y2 * width + x2;              // bottom-right
  
  int out_base = (y * width + x) * channel;
  
  for (int c = 0; c < channel; c++) {
    long *S = S_device[c];

    long sum = S[y2 * width + x2];
    if (x1 > 0) sum -= S[y2 * width + (x1 - 1)];
    if (y1 > 0) sum -= S[(y1 - 1) * width + x2];
    if (x1 > 0 && y1 > 0) sum += S[(y1 - 1) * width + (x1 - 1)];
    
    int idx = (y * width + x) * channel + c;
    output[idx] = (unsigned char)(sum / area);
  }
}



void blur_image(unsigned char *in_image, unsigned char *out_image, int width, int height, int channel, int kernel) {
  // Allocate device memory
  unsigned char *din_image, *dout_image;
  cudaMalloc(&din_image, width * height * channel * sizeof(unsigned char));
  cudaMalloc(&dout_image, width * height * channel * sizeof(unsigned char));

  long **d_s = (long **)malloc(channel * sizeof(long *));
  long **h_S_ptrs = (long **)malloc(channel * sizeof(long *));
  long **d_S_ptrs;
  long *d_temp;

  cudaMalloc(&d_temp, width * height * sizeof(long));
  cudaMalloc(&d_S_ptrs, channel * sizeof(long *));

  for (int c = 0; c < channel; c++) {
    cudaMalloc(&d_s[c], width * height * sizeof(long));
    h_S_ptrs[c] = d_s[c];
  }

  cudaMemcpy(din_image, in_image, width * height * channel * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_S_ptrs, h_S_ptrs, channel * sizeof(long *), cudaMemcpyHostToDevice);

  int blockSizeH = 1;
  while (blockSizeH < width) blockSizeH <<= 1;
  if (blockSizeH > 1024) blockSizeH = 1024; // if width>1024 you'll need chunking / multi-block scan
  dim3 gridH(height);
  size_t sharedBytesH = blockSizeH * sizeof(long);

  // vertical: one block per column, blockDim.x must be >= height
  int blockSizeV = 1;
  while (blockSizeV < height) blockSizeV <<= 1;
  if (blockSizeV > 1024) blockSizeV = 1024; // if height>1024 you'll need chunking / multi-block scan
  dim3 gridV(width);
  size_t sharedBytesV = blockSizeV * sizeof(long);

  // Compute integral for 3 channels
  for (int c = 0; c < channel; c++) {
    horizontal_integral<<<gridH, blockSizeH, sharedBytesH>>>(din_image, d_temp, width, height, channel, c);
    vertical_integral<<<gridV, blockSizeV, sharedBytesV>>>(d_temp, d_s[c], width, height);
  }

  int r = (kernel - 1) / 2;
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  blur_from_integral<<<numBlocks, threadsPerBlock>>>(d_S_ptrs, dout_image, width, height, channel, r);
  cudaMemcpy(out_image, dout_image, width * height * channel * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
  // Free device memory
  cudaFree(din_image);
  cudaFree(dout_image);
  cudaFree(d_temp);
  cudaFree(d_S_ptrs);
  for (int c = 0; c < channel; c++) {
    cudaFree(d_s[c]);
  }
  
  free(d_s);
  free(h_S_ptrs);
}

int main() {
  // Input and Output file name
	const char *input_path = "input.png";
	const char *output_path = "blurred_output.png";

	// kernel size
	int kernel = 15;  

	// Read an input image
	int width, height, channel;
	unsigned char *in_image = stbi_load(input_path, &width, &height, &channel, 0);

	// Read image error
	if (!in_image) {
		printf("Error: Cannot load image %s\n", input_path);
		return -1;
	}

	printf("Image loaded: %dx%d (%d channels)\n", width, height, channel);

  unsigned char *out_image = (unsigned char *)malloc((size_t)width * height * channel * sizeof(unsigned char));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Compute blur image
  cudaEventRecord(start);
  blur_image(in_image, out_image, width, height, channel, kernel);
  cudaEventRecord(stop);
  
  stbi_write_png(output_path, width, height, channel, out_image, width * channel);
  stbi_image_free(in_image);
  free(out_image);
  return 0;
}