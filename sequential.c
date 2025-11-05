#include <stdio.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

void blurImageRGB(unsigned char *in_image, unsigned char *out_image, int width, int height, int channel, int kernel) {
	int r = (kernel - 1) / 2;

	// Allocate integral image (channel channels)
	long ***integral_img_arr = (long ***)malloc(channel * sizeof(long **));
	for (int c = 0; c < channel; c++) {
		integral_img_arr[c] = (long **)malloc(height * sizeof(long *));
		for (int y = 0; y < height; y++)
			integral_img_arr[c][y] = (long *)calloc(width, sizeof(long));
	}

	// Step 1: Build integral images
	for (int c = 0; c < channel; c++) {
		for (int y = 0; y < height; y++) {
			long rowSum = 0;
			for (int x = 0; x < width; x++) {
				int idx = (y * width + x) * channel + c;
				rowSum += in_image[idx];
				if (y == 0)
					integral_img_arr[c][y][x] = rowSum;
				else
					integral_img_arr[c][y][x] = integral_img_arr[c][y - 1][x] + rowSum;
			}
		}
	}


	// Validate integral image
	printf("Integral image sums (bottom-right corner):\n");
	for (int c = 0; c < channel; c++) {
		printf("Channel %d: %ld\n", c, integral_img_arr[c][height-1][width-1]);
	}

	// Step 2: Compute blurred image
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int x1 = MAX(0, x - r);
			int y1 = MAX(0, y - r);
			int x2 = MIN(width - 1, x + r);
			int y2 = MIN(height - 1, y + r);
			int area = (x2 - x1 + 1) * (y2 - y1 + 1);

			for (int c = 0; c < channel; c++) {
				long sum = integral_img_arr[c][y2][x2];
				if (x1 > 0) sum -= integral_img_arr[c][y2][x1 - 1];
				if (y1 > 0) sum -= integral_img_arr[c][y1 - 1][x2];
				if (x1 > 0 && y1 > 0) sum += integral_img_arr[c][y1 - 1][x1 - 1];

				int idx = (y * width + x) * channel + c;
				out_image[idx] = (unsigned char)(sum / area);
			}
		}
	}

	// Free memory
	for (int c = 0; c < channel; c++) {
		for (int y = 0; y < height; y++)
			free(integral_img_arr[c][y]);
		free(integral_img_arr[c]);
	}
	free(integral_img_arr);
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

	// Print first pixel
	int x = 0, y = 0;
	int idx = (y * width + x) * channel;
	printf("Pixel at (%d,%d): ", x, y);
	for (int c = 0; c < channel; c++) {
		printf("%d ", in_image[idx + c]);
	}
	printf("\n");

	// Allocate output image array
	unsigned char *out_image = (unsigned char *)malloc(width * height * channel);

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
