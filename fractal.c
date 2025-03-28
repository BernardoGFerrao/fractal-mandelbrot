#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define LARGURA 1920
#define ALTURA 1080
#define MAX_ITERACOES 5000

int calcular_mandelbrot(double c_real, double c_imag) {
    double z_real = 0.0, z_imag = 0.0;
    int iteracoes = 0;
    while (iteracoes < MAX_ITERACOES) {
        double z_real_quadrado = z_real * z_real;
        double z_imag_quadrado = z_imag * z_imag;

        if (z_real_quadrado + z_imag_quadrado > 4.0) {
            break;
        }

        z_imag = 2 * z_real * z_imag + c_imag;
        z_real = z_real_quadrado - z_imag_quadrado + c_real;
        iteracoes++;
    }
    return iteracoes;
}

int main(int argc, char** argv) {
    int rank, num_processos;
    int num_threads = 1;
    int* buffer_resultado;

    double tempo_inicial, tempo_final, tempo_total;
    double tempo_total_max;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processos);

    // Determina o número de threads OpenMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    if (rank == 0) {
        printf("Executando com %d processos MPI e %d threads OpenMP por processo.\n", num_processos, num_threads);
    }

    tempo_inicial = MPI_Wtime();

    //ZOOM 1
    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;

    //ZOOM 2
//    double x_min = -0.75, x_max = -0.74;
//    double y_min = 0.1, y_max = 0.11;

    //ZOOM 3
//    double x_min = -0.7449, x_max = -0.742;
//    double y_min = 0.099, y_max = 0.10;

    //ZOOM 4
//    double x_min = -0.7439, x_max = -0.745;
//    double y_min = 0.099, y_max = 0.10;

    int linhas_por_processo = ALTURA / num_processos;
    int linha_inicial = rank * linhas_por_processo;
    int linha_final = (rank == num_processos - 1) ? ALTURA : linha_inicial + linhas_por_processo;

    int local_size = (linha_final - linha_inicial) * LARGURA;
    buffer_resultado = (int*)malloc(local_size * sizeof(int));

    #pragma omp parallel for collapse(2)
    for (int y = linha_inicial; y < linha_final; y++) {
        for (int x = 0; x < LARGURA; x++) {
            double c_real = x_min + (x_max - x_min) * x / LARGURA;
            double c_imag = y_min + (y_max - y_min) * y / ALTURA;
            buffer_resultado[(y - linha_inicial) * LARGURA + x] = calcular_mandelbrot(c_real, c_imag);
        }
    }

    tempo_final = MPI_Wtime();
    tempo_total = tempo_final - tempo_inicial;
    MPI_Reduce(&tempo_total, &tempo_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int* resultado_global = (int*)malloc(LARGURA * ALTURA * sizeof(int));
        MPI_Gather(buffer_resultado, local_size, MPI_INT, resultado_global, local_size, MPI_INT, 0, MPI_COMM_WORLD);
        FILE* arquivo_saida = fopen("mandelbrot.ppm", "wb");
        fprintf(arquivo_saida, "P6\n%d %d\n255\n", LARGURA, ALTURA);
        for (int i = 0; i < LARGURA * ALTURA; i++) {
            unsigned char cor = (resultado_global[i] * 255) / (MAX_ITERACOES / 5);
            fputc(cor, arquivo_saida);
            fputc(cor, arquivo_saida);
            fputc(cor, arquivo_saida);
        }
        fclose(arquivo_saida);
        printf("Tempo de execucao total: %f segundos\n", tempo_total_max);
        free(resultado_global);
    } else {
        MPI_Gather(buffer_resultado, local_size, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    free(buffer_resultado);
    MPI_Finalize();
    return 0;
}
