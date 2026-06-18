#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

struct matrix{
    int lin;
    int col;
    Matrix *right;
    Matrix *bellow;
    float info;
};

static int matrix_aloca_cabecas(Matrix ** mc, int i, int j){ // número de linhas e colunas comecando em 1
    //Alocando cabeça das cabeças
    if (i < 1 || j < 1) return 1;

    (*mc) = (Matrix*) malloc(sizeof(Matrix));
    if((*mc) == NULL) return 2;

    (*mc)->lin = i;
    (*mc)->col = j;
    (*mc)->info = 0;

    Matrix* aux = (*mc); //auxiliar para percorrer matriz

    //Alocando cabeças colunas e acertando ponteiros
    for (int k = 0; k < j; k++){
        aux->right = (Matrix*)malloc(sizeof(Matrix));
        if (aux->right == NULL) return 3;
        
        aux->right->lin = -1;
        aux->right->col = k;
        aux = aux-> right;
        aux->bellow = aux;
    }
    aux -> right = *mc;
    
    //Alocando cabeças linhas e acertando ponteiros
    aux = *mc;
    for (int k = 0; k < i; k++){
        aux->bellow = (Matrix*)malloc(sizeof(Matrix));
        if (aux-> bellow == NULL) return 4;
        aux->bellow->col = -1;
        aux->bellow->lin = k;
        aux = aux-> bellow;
        aux->right = aux;
    }
    aux -> bellow = *mc;

    return 0;
}

//funcao para mover ponteiro para uma posição antes da desejada na linha
static Matrix* search_lin(const Matrix* m, int x, int y){
    
    Matrix* aux = m->bellow;

    //descendo nas cabecas linhas
    for(int i = 1; i < x; i++){
        aux = aux->bellow;
    }

    //procurando elemento antes da coluna desejada
    while(((aux->right)->col < y - 1) && ((aux->right)->col != -1)){
        aux = aux->right;
    }

    return aux;
}

//funcao para mover ponteiro para uma posição antes da desejada na coluna
static Matrix* search_col(const Matrix* m, int x, int y){
    Matrix* aux = m->right;

    //deslocando pelas cabecas colunas
    for(int i = 1; i < y; i++){
        aux = aux->right;
    }

    //procurando elemento antes da linha desejada
    while(((aux->bellow)->lin < x - 1) && ((aux->bellow)->lin != -1)){
        aux = aux->bellow;
    }
    
    return aux;
}

int matrix_setelem( Matrix* m, int x, int y, float elem ){
    if (x > m->lin || y > m->col || x < 1 || y < 1){
        return 5;
    }

    //procura posição
    Matrix* aux = search_lin(m, x, y);
    
    if((aux->right->col == y - 1)){//coordenada ja cadastrada
        if(elem != 0) aux->right->info = elem;
        else{
            Matrix* aux2 = search_col(m, x, y);
            Matrix* aux3 = aux->right->right;//salvando aonde aponta na linha
            Matrix* aux4 = aux2->bellow->bellow;//salvando aonde aponta na coluna
            free(aux->right);
            aux->right = aux3;//ajustando ponteiros na linha
            aux2->bellow = aux4;//ajustando ponteiros na coluna
        }
    }else{//coordenada não cadastrada
        if (elem != 0){
            //alocando
            Matrix* el = (Matrix*)malloc(sizeof(Matrix));
            if(el == NULL) return 6;
            el->info = elem;
            el->lin = x - 1;
            el->col = y - 1;
            //acertando ponteiros linha
            el->right = aux->right;
            aux->right = el;
            //acertando ponteiros coluna
            Matrix *aux2 = search_col(m, x, y);
            el->bellow = aux2->bellow;
            aux2->bellow = el;
        }
    }
    return 0;
}

int matrix_create( Matrix** mc ){
    //Alocando cabeças com as informações
    int i, j;
    scanf("%d %d", &i, &j);
    if (matrix_aloca_cabecas(mc, i, j) != 0) return 7;

    //alocando elementos da matriz
    float val;
    Matrix *aux = *mc;
    while(1){
        scanf("%d", &i);
        if (i == 0) break;
        
        scanf(" %d %f", &j, &val);
        if(matrix_setelem(*mc, i, j, val) != 0) return 8;
    }
    return 0;
}

int matrix_print( const Matrix* m ){
    Matrix* aux = m->bellow;
    printf("%d %d\n", m->lin, m->col);
    
    while(aux->lin != m->lin){
        while(aux->right->col != -1){
            printf("%d %d %.3f\n", ((aux->right->lin)+1), ((aux->right->col)+1), (aux->right->info));
            aux = aux->right;
        }
        aux = aux->right;
        aux = aux->bellow;
    }
    printf("0\n");
   return 0;
}

int matrix_getelem( const Matrix* m, int x, int y, float *elem ){
    if (x > m->lin || y > m->col || x < 1 || y < 1) return 9;

    Matrix *aux = search_lin(m, x, y);

    if(aux->right->col == y - 1) *elem = aux->right->info;
    else *elem = 0.0;

    return 0;
}

int matrix_transpose( const Matrix* m, Matrix** r ){
    if(matrix_aloca_cabecas(r, m->col, m->lin)!=0) return 10;

    Matrix* aux = m->bellow;
    
    while(aux->lin != m->lin){
        while(aux->right->col != -1){
            if(matrix_setelem(*r, aux->right->col + 1, aux->right->lin + 1, aux->right->info) != 0)
                return 11;
            
            aux = aux->right;
        }
        aux = aux->right;
        aux = aux->bellow;
    }
    return 0;
}

int matrix_destroy( Matrix* m ){
    //desalocando elementos
    if (m == NULL) return 30;
    else{
    Matrix* aux = m->bellow;
    Matrix* aux2;
    
    while (aux->lin != m->lin){
        while(aux->right->col != -1){
            aux2 = aux->right->right;
            free(aux->right);
            aux->right = aux2;
        }
        aux = aux->bellow;
    }
    
    //desalocando cabecas colunas
    while (m->right != m){
        aux = m->right->right;
        free(m->right);
        m->right = aux;
    }

    //desalocando cabecas linhas
    while (m->bellow != m){
        aux = m->bellow->bellow;
        free(m->bellow);
        m->bellow = aux;
    }
    
    free(m);
    //usuario responsavel por m = NULL
    //se for feito destroyers seguidos sem fazer apontar para null, pode ter acesso
    //a area ja desalocada
    return 0;
    }
}

int matrix_add( const Matrix* m, const Matrix* n, Matrix** r ){
    if (m->lin != n->lin || m->col != n->col){
        printf("ERRO! Número de linhas ou colunas não compatíveis para a soma.\n");
        return 12;
    }    
    
    if(matrix_aloca_cabecas(r, m->lin, m->col) != 0) return 13;

    Matrix *aux1 = m->bellow;
    Matrix *aux2 = n->bellow;

    for(int i = 1; i <= m->lin; i++){
        while (aux1->right->col != -1 || aux2->right->col != -1){
            if(aux1->right->col == -1){
                if(matrix_setelem(*r, i, aux2->right->col + 1, aux2->right->info) != 0) return 25;
                aux2 = aux2->right;
            }else if(aux2->right->col == -1){
                if(matrix_setelem(*r, i, aux1->right->col + 1, aux1->right->info) != 0) return 26;
                aux1 = aux1->right;
            }else if (aux1->right->col == aux2->right->col){
                if(matrix_setelem(*r, i, aux1->right->col + 1, aux1->right->info + aux2->right->info) != 0) return 25;
                aux1 = aux1->right;
                aux2 = aux2->right;
            }else if(aux1->right->col > aux2->right->col){
                if(matrix_setelem(*r, i, aux2->right->col + 1, aux2->right->info) != 0) return 27;
                aux2 = aux2->right;
            }else if(aux1->right->col < aux2->right->col){
                if(matrix_setelem(*r, i, aux1->right->col + 1, aux1->right->info) != 0) return 28;
                aux1 = aux1->right;
            }else{
                return 26;
            }
        }
        aux1 = aux1->right->bellow;
        aux2 = aux2->right->bellow;
    }
    return 0;
}

int matrix_multiply( const Matrix* m, const Matrix* n, Matrix** r ){
    if (m->col != n->lin) {
        printf("ERRO! Número de linhas ou colunas não compatíveis para a multiplicação.\n");
        return 17;
    }

    Matrix* t;
    if(matrix_transpose(n, &t) != 0) return 32;/*usando transposta para facilitar a multiplicação*/
    
    if(matrix_aloca_cabecas(r, m->lin, n->col) != 0) return 18;

    Matrix *aux1 = m->bellow;
    Matrix *aux2 = t->bellow;
    float p;

    for(int i = 1; i <= m->lin; i++){
        for (int j = 1; j <= t->lin; j++){
            p = 0;
            while (aux1->right->col != -1 && aux2->right->col != -1){
                if (aux1->right->col == aux2->right->col){
                    p += (aux1->right->info) * (aux2->right->info);
                    aux1 = aux1->right;
                    aux2 = aux2->right;
                }else if(aux1->right->col < aux2->right->col){
                    aux1 = aux1->right;
                }else if(aux1->right->col > aux2->right->col){
                    aux2 = aux2->right;
                }else return 35;
            }
            if (matrix_setelem(*r, i, j, p) != 0) return 37;
            while(aux1->col != -1) aux1 = aux1->right;
            while(aux2->col != -1) aux2 = aux2->right;
            aux2 = aux2->bellow;
        }
        aux2 = aux2->bellow;//below a mais para descer da cabeca das cabecas
        aux1 = aux1->bellow;
    }
    matrix_destroy(t);
    return 0;
}