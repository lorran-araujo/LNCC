#include <stdio.h>
#include "matrix.h"

int main( void ) {
/* Inicializacao da aplicacao ... */
    Matrix *A=NULL;
    Matrix *B=NULL;
    Matrix *C=NULL;

    if( !matrix_create( &A ) ){
        printf("\nPrintando a Matriz A:\n");
        matrix_print( A );
    }else {
        fprintf( stderr, "Erro na alocacao de A como listas encadeadas.\n" );
        return 1;
    }

    if( !matrix_create( &B ) ){
        printf("\nPrintando a Matriz B:\n");
        matrix_print( B );
    }else {
        fprintf( stderr, "Erro na alocacao de B como listas encadeadas.\n" );
        return 1;
    }

    if ( !matrix_add( A, B, &C ) ) {
        printf("\nPrintando a Matriz A+B:\n");
        matrix_print( C );
    }
    else
        fprintf( stderr, "Erro na soma C=A+B.\n" );
    matrix_destroy( C );

    if ( !matrix_multiply( A, B, &C ) ){
        printf("\nPrintando a Matriz AB:\n");
        matrix_print( C );
    }else
        fprintf( stderr, "Erro na multiplicacao C=A*B.\n" );
        matrix_destroy( C );

    if ( !matrix_transpose( A, &C ) ){
        printf("\nPrintando a transposta da Matriz A:\n");
        matrix_print( C );
    }else
        fprintf( stderr, "Erro na transposicao C=A^T.\n" );

    for(int i = 1; i <= 100; i++){
        for(int j = 1; j <= 100; j++){
            matrix_setelem(A, i, j, 0);
        }
    }

    matrix_print(A);

    if(A != NULL) matrix_destroy( A );
    if(B != NULL) matrix_destroy( B );
    if(C != NULL) matrix_destroy( C );

    return 0;
}