#include <stdio.h>
#include <cuda.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Sequential scan per row (one thread per row)
__global__ void horizontal_integral(unsigned char *input, long long *output, int width, int height, int C, int channel) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row >= height) return;
  
  // Each thread processes one complete row sequentially
  long long sum = 0;
  for (int col = 0; col < width; col++) {
    int idx = (row * width + col) * C + channel;
    sum += input[idx];
    output[row * width + col] = sum;
  }
}

// Sequential scan per column (one thread per column)
__global__ void vertical_integral(long long *input, long long *output, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (col >= width) return;
  
  // Each thread processes one complete column sequentially
  long long sum = 0;
  for (int row = 0; row < height; row++) {
    sum += input[row * width + col];
    output[row * width + col] = sum;
  }
}

__global__ void blur_from_integral(long long **S_device, unsigned char *output, int width, int height, int channel, int r) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= width || y >= height) return;
  
  int x1 = max(0, x - r);
  int y1 = max(0, y - r);
  int x2 = min(width - 1, x + r);
  int y2 = min(height - 1, y + r);
  
  int area = (x2 - x1 + 1) * (y2 - y1 + 1);
  
  for (int c = 0; c < channel; c++) {
    long long *S = S_device[c];

    long long sum = S[y2 * width + x2];
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

  long long **d_s = (long long **)malloc(channel * sizeof(long long *));
  long long **h_S_ptrs = (long long **)malloc(channel * sizeof(long long *));
  long long **d_S_ptrs;
  long long *d_temp;

  cudaMalloc(&d_temp, width * height * sizeof(long long));
  cudaMalloc(&d_S_ptrs, channel * sizeof(long long *));

  for (int c = 0; c < channel; c++) {
    cudaMalloc(&d_s[c], width * height * sizeof(long long));
    h_S_ptrs[c] = d_s[c];
  }

  cudaMemcpy(din_image, in_image, width * height * channel * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_S_ptrs, h_S_ptrs, channel * sizeof(long long *), cudaMemcpyHostToDevice);

  // Grid configuration: one thread per row for horizontal scan
  int threadsPerBlockH = (height <= 1024) ? height : 1024;
  int numBlocksH = (height + threadsPerBlockH - 1) / threadsPerBlockH;

  // Grid configuration: one thread per column for vertical scan
  int threadsPerBlockV = (width <= 1024) ? width : 1024;
  int numBlocksV = (width + threadsPerBlockV - 1) / threadsPerBlockV;

  // Compute integral for all channels
  for (int c = 0; c < channel; c++) {
    horizontal_integral<<<numBlocksH, threadsPerBlockH>>>(din_image, d_temp, width, height, channel, c);
    cudaDeviceSynchronize();
    vertical_integral<<<numBlocksV, threadsPerBlockV>>>(d_temp, d_s[c], width, height);
    cudaDeviceSynchronize();
  }

  int r = (kernel - 1) / 2;
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  blur_from_integral<<<numBlocks, threadsPerBlock>>>(d_S_ptrs, dout_image, width, height, channel, r);
  cudaDeviceSynchronize();
  
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


  // Compute blur image
  clock_t start = clock();
  blur_image(in_image, out_image, width, height, channel, kernel);
  clock_t end = clock();

  double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Blurring done in %.4f seconds (kernel=%d)\n", elapsed, kernel);

  stbi_write_png(output_path, width, height, channel, out_image, width * channel);
  stbi_image_free(in_image);
  free(out_image);
  return 0;
}