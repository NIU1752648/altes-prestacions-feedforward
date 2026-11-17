/**
 *  main.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de
 * l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona Created on: 31
 * gen. 2019 Last modified: fall 24 (curs 24-25) Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funció que entrena la xarxa neuronal definida + Funció que fa el test del
 * model entrenat + programa principal.
 *
 *  @todo: modificar funció d'entrenament per utilitzar epochs i batchs!!!
 *        Epochs, fet (23/01/25)
 *        Batchs implica canviar les funcions de back_propagation i
 * update_weights de training.c, de moment ho deixem com està Revisar
 * print_recognized
 */

#include "main.h"

#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <mpi.h>

// MPI: identificador d'aquest procés (0..size-1) i nombre total de processos
int rank = 0;
int size = 1;

/* PROFILE: instrumentació de temps per procés (compilar amb -DPROFILE).
 * Fases que es mesuren:
 *   load    càrrega del dataset d'entrenament
 *   compute feed_input + forward_prop + compute_loss + back_prop dels patrons propis
 *   wait    MPI_Barrier abans de la reducció: temps esperant els processos més lents
 *   comm    MPI_Allreduce dels gradients (amb tots ja sincronitzats: cost de la comunicació)
 *   update  init_deltas + update_weights (feina repetida a tots els processos)
 *   test    test_nn complet
 * Sense -DPROFILE les macros queden buides i el codi és idèntic a la versió normal. */
#ifdef PROFILE
enum { T_LOAD, T_COMPUTE, T_WAIT, T_COMM, T_UPDATE, T_TEST, T_NUM };
static const char* t_names[T_NUM] = {"load", "compute", "wait", "comm", "update", "test"};
static double t_acc[T_NUM] = {0};
#define TIC(v) double v = MPI_Wtime()
#define TOC(fase, v) t_acc[fase] += MPI_Wtime() - (v)
#else
#define TIC(v)
#define TOC(fase, v)
#endif

//-----------FREE INPUT------------
void freeInput(int np, char** input) {
    for (int i = 0; i < np; i++)
        free(input[i]);
    free(input);
}

//-----------PRINTRECOGNIZED------------
void printRecognized(int p, layer Output) {
    int imax = 0;

    for (int i = 1; i < num_out_layer; i++)
        if (Output.actv[i] > Output.actv[imax])
            imax = i;

    if (imax == Validation[p])
        total++;

    if (debug == 2) {
        printf("Patró %d sembla %c i és %d", p, '0' + imax,
               Validation[p]);
        for (int k = 0; k < num_out_layer; k++)
            printf("   %f", Output.actv[k]);
        printf("\n");
    }
}

/**
 * @brief Entrena la xarxa neuronal en base al conjunt d'entrenament
 *
 * @details Primer carrega tots els patrons d'entrenament (loadPatternSet)
 *          Després realitza num_epochs iteracions d'entrenament.
 *          Cada epoch fa:
 *              - Determina aleatòriament l'ordre en que es consideraran els
 * patrons (per evitar overfitting)
 *              - Per cada patró d'entrenament fa el forward_prop (reconeixament
 * del patró pel model actual) i el back_prop i update_weights (ajustament de
 * pesos i biaxos per provar de millorar la precisió del model)
 *
 * @see loadPatternSet, feed_input, forward_prop, back_prop, update_weights,
 * freeInput
 *
 */
void train_neural_net() {
    if (rank == 0) printf("\nTraining...\n");

    TIC(t_load);
    if ((input = loadPatternSet(num_training_patterns, dataset_training_path,
                                1)) == NULL) {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }
    TOC(T_LOAD, t_load);

    int ranpat[num_training_patterns];

    for (int p = 0; p < num_training_patterns; p++)
        ranpat[p] = p;

    // Gradient Descent
    for (int it = 0; it < num_epochs; it++) {
        float loss = 0;
        // Train patterns randomly
        for (int p = 0; p < num_training_patterns; p++) {
            int x = rando();
            int np = (x * x) % num_training_patterns;
            int op = ranpat[p];
            ranpat[p] = ranpat[np];
            ranpat[np] = op;
        }

        // MPI: tots els processos tenen la mateixa llavor, per tant ranpat[]
        // queda igual a tots sense comunicar res.

        for (int nb = 0; nb < num_training_patterns/batch_size; nb++) {
            TIC(t_init);
            init_deltas();
            TOC(T_UPDATE, t_init);

            // MPI: repartiment cíclic dels patrons del batch. El procés 'rank'
            // fa els patrons rank, rank+size, rank+2*size... del batch.
            TIC(t_comp);
            for (int i = nb*batch_size + rank; i < (nb+1)*batch_size; i += size) {
                int p = ranpat[i];

                feed_input(p);
                forward_prop();
                loss += compute_loss(p);   // loss local d'aquest procés
                back_prop(p);              // gradient parcial local
            }
            TOC(T_COMPUTE, t_comp);

#ifdef PROFILE
            // Separa l'espera pels processos lents del cost real de l'Allreduce
            TIC(t_wait);
            MPI_Barrier(MPI_COMM_WORLD);
            TOC(T_WAIT, t_wait);
#endif
            TIC(t_comm);

            // MPI: suma dels gradients parcials de tots els processos.
            // dw de la capa l té num_neurons[l+1] x num_neurons[l] elements.
            // back_prop escriu els dbias de les capes 1..num_layers-1, que
            // tenen num_neurons[l+1] elements.
            for (int l = 0; l < num_layers - 1; l++) {
                int nw = num_neurons[l + 1] * num_neurons[l];
                int nbias = num_neurons[l + 1];
                MPI_Allreduce(MPI_IN_PLACE, lay[l].dw, nw, MPI_FLOAT,
                              MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, lay[l + 1].dbias, nbias, MPI_FLOAT,
                              MPI_SUM, MPI_COMM_WORLD);
            }
            TOC(T_COMM, t_comm);

            // Tots els processos tenen el mateix gradient: apliquen el mateix canvi
            TIC(t_upd);
            update_weights();
            TOC(T_UPDATE, t_upd);
        }

        // MPI: loss de tot l'epoch (una reducció per epoch, només per imprimir)
        float global_loss = 0.0;
        MPI_Reduce(&loss, &global_loss, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
            printf("Epoch %d Loss: %f\n", it, global_loss/num_training_patterns);
        if (debug == 1 && rank == 0) getchar();
    }
    freeInput(num_training_patterns, input);
}

//-----------TEST THE TRAINED NETWORK------------
void test_nn() {
    char** rSet;

    printf("\nTesting...\n");

    if ((rSet = loadPatternSet(num_test_patterns, dataset_test_path, 0)) ==
        NULL) {
        printf("Error!!\n");
        exit(-1);
    }

    for (int i = 0; i < num_test_patterns; i++) {
        for (int j = 0; j < num_neurons[0]; j++)
            lay[0].actv[j] = rSet[i][j];

        forward_prop();
        printRecognized(i, lay[num_layers - 1]);
    }

    printf("\nTotal encerts = %d (/%d) \n", total, num_test_patterns);
    freeInput(num_test_patterns, rSet);
}

#ifdef PROFILE
/* Recull els temps de tots els processos al rank 0 i imprimeix, per fase,
 * el mínim, la mitjana i el màxim entre processos (en segons).
 * Format pensat per filtrar amb: grep '^PROFILE' */
static void print_profile(void) {
    double t_min[T_NUM], t_max[T_NUM], t_sum[T_NUM];
    MPI_Reduce(t_acc, t_min, T_NUM, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(t_acc, t_max, T_NUM, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(t_acc, t_sum, T_NUM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("PROFILE P=%d\n", size);
        printf("PROFILE %-8s %10s %10s %10s\n", "fase", "min", "avg", "max");
        for (int f = 0; f < T_NUM; f++)
            printf("PROFILE %-8s %10.4f %10.4f %10.4f\n", t_names[f], t_min[f],
                   t_sum[f] / size, t_max[f]);
    }
}
#endif

//-----------MAIN-----------//
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (debug == 1)
        printf("argc = %d \n", argc);
    if (argc <= 1)
        readConfiguration("configuration/configfile.txt");
    else
        readConfiguration(argv[1]);

    if (debug == 1)
        printf("FINISH CONFIG \n");

    // Initialize the neural network module
    if (init() != SUCCESS_INIT) {
        printf("Error in Initialization...\n");
        exit(0);
    }

    // Start measuring time
    struct timeval begin;
    gettimeofday(&begin, 0);

    // Train
    train_neural_net();

    // Test
    TIC(t_test);
    test_nn();
    TOC(T_TEST, t_test);

    // Stop measuring time and calculate the elapsed time
    double elapsed = elapsedT(begin);

    if (dinit() != SUCCESS_DINIT)
        printf("Error in Dinitialization...\n");

    printf("\n\nGoodbye! (%f sec)\n\n", elapsed);

#ifdef PROFILE
    print_profile();
#endif

    MPI_Finalize();
    return 0;
}