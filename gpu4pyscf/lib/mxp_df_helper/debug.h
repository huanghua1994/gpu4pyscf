#pragma once

template <typename T>
std::vector<T> dev_array_to_host_array(const T* dev_array, size_t size) 
{
    std::vector<T> host_array(size);
    cudaMemcpy(host_array.data(), dev_array, sizeof(T) * size, cudaMemcpyDeviceToHost);
    return host_array;
}

template <typename T>
void dump_binary(const char *filename, const T* data, size_t size)
{
    FILE *ouf = fopen(filename, "wb");
    if (!ouf)
    {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        return;
    }
    size_t written = fwrite(data, sizeof(T), size, ouf);
    if (written != size)
        fprintf(stderr, "Error writing to file %s: expected %zu, wrote %zu\n", filename, size, written);
    fclose(ouf);
}

template <typename T>
void print_rm_matrix(const T *mat, const int ldm, const int nrow, const int ncol, const char *name)
{
    printf("\nMatrix %s (top-left %d x %d):\n", name, nrow, ncol);
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < ncol; j++)
            printf("% 10.8f ", static_cast<double>(mat[i * ldm + j]));
        printf("\n");
    }
    printf("\n");
}
