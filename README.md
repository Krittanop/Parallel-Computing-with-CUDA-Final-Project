# Parallel Computing with CUDA - Image Blur

Fast box blur implementation using integral images, with both sequential (CPU) and parallel (CUDA GPU) versions.

## Overview

This project implements efficient box blur using the **integral image algorithm**. The parallel version uses CUDA to accelerate integral image construction.

---

##  Quick Start

### Sequential (CPU) Version

**Compile and run:**
```bash
gcc sequential.c -o sequential
./sequential
```

**Expected output:**
```
Image loaded: 4000x3000 (3 channels)
Blurring done in 0.xxxx seconds (kernel=30)
```

### Parallel (CUDA GPU) Version

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed

**Compile and run:**
```bash
nvcc parallel.cu -o parallel
./parallel
```

**Expected output:**
```
Image loaded: 4000x3000 (3 channels)
Blurring done in 0.xxxx seconds (kernel=30)
```

---

## Configuration

Edit the `main()` function in either `sequential.c` or `parallel.cu`:

```c
// Input and Output file names
const char *input_path = "input.png";           // Input image
const char *output_path = "blurred_output.png"; // Output image

// Kernel size (must be odd number)
int kernel = 30;  // Blur radius = (kernel-1)/2
```


## Algorithm Details

### Integral Image Method

- **Build Integral Image** - Two-pass prefix sum:
   - Horizontal scan: Row-wise cumulative sum
   - Vertical scan: Column-wise cumulative sum


**Work and Span Analysis:**
- **Work:** O(W × H × C)
- **Span:** O(W + H)
- **Parallelism:** W×H/(W+H)

---

##  Performance Comparison

### Time Complexity

| Method | Work | Span |
|--------|------------|-----------------|
| **Sequential Solution** | O(W×H×C) | O(W×H×C) |
| **Parallel Solution** | O(W×H×C) | O((W+H)×C) |
|  | Work Efficient | Degree of parallelism = O(W×H) / O(W+H)  |


## Implementation Versions

### Current: Sequential Scan (Simple & Efficient)
- One thread per row/column
- Sequential scan within each row/column
- Work-efficient O(n)
- Good parallelism across rows/columns
- No shared memory

---