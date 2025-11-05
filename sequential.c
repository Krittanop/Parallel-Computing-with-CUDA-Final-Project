#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

void blurImageRGB(unsigned char *in_image, unsigned char *out_image, int width, int height, int channel, int kernel) {
	int r = (kernel - 1) / 2;

	int integral_img_size = (int)width * height;
	int total_elements = integral_img_size * channel;
	long *integral_img = (long *)calloc(total_elements, sizeof(long));

	// Step 1: Build integral images
	for (int c = 0; c < channel; c++) {
		int channel_offset = integral_img_size * c;
		
		for (int y = 0; y < height; y++) {
			long rowSum = 0;
			int row_offset = channel_offset + (int)y * width;
			
			for (int x = 0; x < width; x++) {
				int idx_in = (y * width + x) * channel + c;
				rowSum += in_image[idx_in];

				int idx_ii = row_offset + x;
				
				if (y == 0) {
					integral_img[idx_ii] = rowSum;
				} else {
					int idx_ii_prev_row = idx_ii - width;
					integral_img[idx_ii] = integral_img[idx_ii_prev_row] + rowSum;
				}
			}
		}
	}


	// Step 2: Compute blurred image
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int x1 = MAX(0, x - r);
			int y1 = MAX(0, y - r);
			int x2 = MIN(width - 1, x + r);
			int y2 = MIN(height - 1, y + r);
			int area = (x2 - x1 + 1) * (y2 - y1 + 1);
			
			int idx_out_base = ((int)y * width + x) * channel;

			for (int c = 0; c < channel; c++) {
				int channel_offset = integral_img_size * c;
				long sum = integral_img[channel_offset + (int)y2 * width + x2];
				if (x1 > 0) sum -= integral_img[channel_offset + (int)y2 * width + (x1 - 1)];

				if (y1 > 0) sum -= integral_img[channel_offset + (int)(y1 - 1) * width + x2];

				if (x1 > 0 && y1 > 0) sum += integral_img[channel_offset + (int)(y1 - 1) * width + (x1 - 1)];

				out_image[idx_out_base + c] = (unsigned char)(sum / area);
			}
		}
	}
	free(integral_img);
}

int main() {
	// Input and Output file name
	const char *input_path = "input.png";
	const char *output_path = "blurred.png";

	// kernel size
	int kernel = 30;

	// Read an input image
	int width, height, channel;
	unsigned char *in_image = stbi_load(input_path, &width, &height, &channel, 0);

	// Read image error
	if (!in_image) {
		printf("Error: Cannot load image %s\n", input_path);
		return -1;
	}

	printf("Image loaded: %dx%d (%d channels)\n", width, height, channel);

	// Allocate output image array
	unsigned char *out_image = (unsigned char *)malloc((size_t)width * height * channel);
	if (!out_image) {
		printf("Error: Failed to allocate memory for output image.\n");
		stbi_image_free(in_image);
		return -1;
	}

	// Compute blur image with timer
	clock_t start = clock();
	blurImageRGB(in_image, out_image, width, height, channel, kernel);
	clock_t end = clock();

	// Convert compute time to second
	double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Blurring done in %.4f seconds (kernel=%d)\n", elapsed, kernel);

	// Write a blur image (Output file)
	stbi_write_png(output_path, width, height, channel, out_image, width * channel);
	printf("Saved blurred image to %s\n", output_path);

	// Free all allocate memory
	stbi_image_free(in_image);
	free(out_image);
	return 0;
}