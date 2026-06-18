#include <stdio.h>

#include "matrix.h"

int main( void ) {
    /* Inicializacao da aplicacao ... */
    Matrix *A=NULL;
    Matrix *B=NULL;
    Matrix *C= NULL;

    if(!matrix_create(&A)){
        //printf("\nPrintando a matriz A:\n");
        //matrix_print( A );
        puts("");
    } else {
        fprintf( stderr, "Erro na alocacao de A como listas encadeadas.\n" );
        puts("");
        return 1;
    }

    if(!matrix_create(&B)){
        //printf("\nPrintando a matriz A:\n");
        //matrix_print( A );
        puts("");
    } else {
        fprintf( stderr, "Erro na alocacao de A como listas encadeadas.\n" );
        puts("");
        return 1;
    }
    
    if(!matrix_add_alt(A, B, &C)){
        printf("\nPrintando a matriz soma:\n");
        matrix_print( C );
        puts("");
    } else {
        fprintf( stderr, "Erro na multiplicação.\n" );
        puts("");
        return 1;
    }

    matrix_destroy( A );
    matrix_destroy( B );
    matrix_destroy( C );
    A = NULL;
    B = NULL;
    C = NULL;
    
    return 0;
}