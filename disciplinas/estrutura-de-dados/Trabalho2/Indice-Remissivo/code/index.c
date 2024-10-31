#include "index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define NPAL 64 //tamanho maximo de cada palavra
#define NTAB 127 //tamanho da tabela de dispersao

//tipo da lista que armazenara as linhas de ocorrencias da palavra
struct ocorrencias{
    int linha; 
    struct ocorrencias* next;
};
typedef struct ocorrencias Oco;

//tipo da lista que salvara as palavras no indice
struct words{
    char palavra[NPAL]; //palavra 
    int n; // numero de ocorrencias
    struct words* next; //proxima palavra no mesmo indice
    struct ocorrencias* oco; //ponteiro para lista de ocorrencias 
};
typedef struct words Words;

//tipo principal
struct index{
    int num; //numero de palavras na tabela
    struct words* Hash[NTAB];//tabela hash
    char* tf; //salva o nome do arquivo text file para uso na put
};

//funcao hash de dispersao da tabela
static int hash (const char* s){
    int i = 0, total = 0;
    while (s[i] != '\0'){
        total += s[i++];
    }

    return total % NTAB;
}

/*cadastra ou atualiza ocorrencias de uma palavra na hash*/
static int update_keys_hash(Index* idx, const char*s, int linha){
    int a = hash(s);
    
    Words* aux = idx->Hash[a];
    //parte do pressusposto que a hash nao estara vazia
    while(strcmp(aux->palavra, s) != 0){
        aux = aux->next;
        if (aux == NULL) return 78;
    }
    
    if(aux->n == 0){ //ocorrencia ainda nao registrada
        aux->oco = (Oco*)malloc(sizeof(Oco));
        if(aux->oco == NULL) return 85;

        aux->n++;

        aux->oco->linha = linha;
        aux->oco->next = NULL; 
        return 0;
    }else{//ja registrada alguma ocorrencia
        Oco *aux3 = aux->oco;
        while(aux3 != NULL){//testa se ocorrencia na linha ja foi registrada
            if (aux3->linha == linha) return 0;
            aux3 = aux3->next;
        }

        Oco* aux2 = aux->oco;
        while(aux2->next != NULL) aux2 = aux2->next;
        aux2->next = (Oco*)malloc(sizeof(Oco));
        if(aux2->next == NULL) return 85;

        aux->n++;

        aux2->next->linha = linha;
        aux2->next->next = NULL;

        return 0;
    }

}

/*le o texto comparando palavra e atualizando indice*/
static int le_text(FILE *tf, const char* k, Index* idx){
    int linha = 1;
    char c;
    char s[NPAL];
    int i = 0;
    
    while(1){
        c = fgetc(tf);
        if (isalpha(c) || c == '-') {
            c = tolower(c); //tranforma em minuscula possiveis letras em maiusculo
            s[i++] = c;
        } else if (i > 0 || (c == EOF && i > 0)){//fim de uma palavra
            s[i] = '\0';
            
            if(strcmp(k, s) == 0){//palavra encontrada no texto
                if(update_keys_hash(idx,k,linha) != 0) return 87;
            }

            // Reinicia o índice da palavra para ler nova palavra
            i = 0;
            if (c == EOF) break;
        }
        else if (c == EOF) break;//fim do arquivo sem palavra no fim

        // Incrementa a linha se o caractere for uma quebra de linha
        if (c == '\n') linha++;
    }

    return 0;
}

//registra palavra na hash
static int register_keys(Index* idx, const char *k){
    int a = hash(k);
    
    if(idx->Hash[a] == NULL){//sem palavras no índice
        idx->Hash[a] = (Words*)malloc(sizeof(Words));
        Words* aux = idx->Hash[a];
        if (aux == NULL) return 12;
        strcpy(aux->palavra, k);
        idx->num++;
        aux->n = 0;
        aux->oco = NULL;
        aux->next = NULL;
        
        return 0;
    }

    
    Words* aux = idx->Hash[a];
    while(aux != NULL){
        if (strcmp(aux->palavra, k) == 0){//palavra ja cadastrada
            return 0;
        }
        aux = aux->next;
    }

    //palavra ainda nao cadastrada e indice nao vazio
    aux = idx->Hash[a];
    Words* aux2 = aux;

    idx->Hash[a] =  (Words*)malloc(sizeof(Words));
    aux = idx->Hash[a];
    if (aux == NULL) return 12;
    strcpy(aux->palavra, k);
    idx->num++;
    aux->n = 0;
    aux->oco = NULL;

    //acertando ponteiros
    aux->next = aux2;

    return 0;
}

//le e cadastra keys
static int le_keys(FILE* kf, FILE *tf, Index* idx){
    char c;
    char s[NPAL];
    int i = 0;
    
    /*while ((c = fgetc(fp)) != EOF) {*/
    while(1){
        c = fgetc(kf);
        if (isalpha(c) || c == '-') {
            c = tolower(c); //tranforma em minuscula possiveis letras em maiusculo
            s[i++] = c;
        } else if (i > 0 || (c == EOF && i > 0)){//fim de uma palavra
            s[i] = '\0';
            
            // Registra palavra na hash
            if (register_keys(idx,s) != 0) return 65;
            
            //Compara palavras no texto e cadastra se necessário
            //if(le_text(tf,s,idx) != 0) return 54;
            if(index_put(idx, s) != 0) return 54;

            // Reinicia o índice da palavra
            i = 0;
            if (c == EOF) break;
        }
        else if (c == EOF) break;//fim do arquivo sem palavra no fim

    }

    return 0;
}

//cria tabela hash
static int start_tab(Index ** idx){
    (*idx) = (Index*)malloc(sizeof(Index));
    if ((*idx) == NULL) return 4;
    for (int i = 0; i < NTAB; i++){
        (*idx)->Hash[i] = NULL;
    }
    (*idx)->num = 0;
    (*idx)->tf = NULL;
    return 0;
}

//funcao para criar arquivo invertido
int index_createfrom (const char *key_file, const char *text_file, Index **idx){
    
    if(start_tab(idx)!= 0) return 12;

    //copia o nome do arquivo com o texto
    int n = strlen(text_file);
    (*idx)->tf = (char*)malloc((n+1)*sizeof(char));
    if((*idx)->tf == NULL) return 65;
    strcpy((*idx)->tf, text_file);

    //abre arquivos para leitura
    FILE* kf = fopen(key_file, "rt");
    if( kf == NULL) return 32;
    FILE* tf = fopen(text_file, "rt");
    if(tf == NULL) return 35;
    
    //le palavras chaves e atualiza tabela hash
    if (le_keys(kf,tf,(*idx)) != 0) return 56;

    //fecha os arquivos
    fclose(kf);
    fclose(tf);

    return 0;
}

//consulta a existência de um índice na tabela
int index_get(const Index *idx, const char *key, int **occurrences, int *num_occurrences){
    char key_min[NPAL];

    int l = 0;
    while(key[l] != '\0'){
        key_min[l] = tolower(key[l]);
        l++;
    }

    key_min[l]='\0';
    
    int i = hash(key_min);
    Words* aux = idx->Hash[i];
    while((aux != NULL)&&(strcmp(aux->palavra, key_min) != 0)){
        aux = aux->next;
    }
    if (aux == NULL) return 18;//nao encontrou a palavra
    
    //encontrou palavra
    *num_occurrences = aux->n;
    if(aux->n > 0){
        *occurrences = (int*)calloc((aux->n),sizeof(int));
        if (*occurrences == NULL) return 18;
        Oco* aux2 = aux->oco;
        for (int i = 0; i < aux->n ; i++){
            (*occurrences)[i] = aux2->linha;
            aux2 = aux2->next;
        }
    }else{
        *occurrences = NULL;    
    }
    return 0; 
}

//reseta as ocorrencias de uma palavra
static int reset_oco(Index* idx, const char* key){
    int i = hash(key);
    
    
    Words* aux = idx->Hash[i];

    while (aux != NULL){
        if(strcmp(aux->palavra, key) == 0) break;
        aux = aux->next;
    }

    if (aux != NULL && strcmp(aux->palavra, key) == 0){ //palavra ja estava registrada
        aux->n = 0; //zera ocorrencias
        
        Oco* aux2;
        while (aux->oco != NULL){
            aux2 = aux->oco->next;
            free(aux->oco);
            aux->oco = aux2;
        }
    }

    return 0;
}

int index_put(Index *idx, const char *key){
    if (idx == NULL) return 26;

    char key_min[NPAL];
    int l = 0;
    while(key[l] != '\0'){
        key_min[l] = tolower(key[l]);
        l++;
    }

    key_min[l]='\0';

    //abre arquivo
    FILE* tf = fopen(idx->tf, "r");
    if( tf == NULL) return 39;

    //reseta ocorrencias caso ja exista
    if(reset_oco(idx, key_min));

    //registra a chave
    if(register_keys(idx, key_min) != 0) return 56;

    //compara chave no texto e cadastra ocorrencias
    if (le_text(tf,key_min,idx) != 0) return 32;
    
    //fecha arquivo
    fclose(tf);

    return 0;
}

//cria vetor que será usado para imprimir indice
static Words** cria_vetor(const Index* idx, int n){
    int i, j = 0;
    Words* p;
    Words **vet = (Words**) malloc(n*sizeof(Words*));

    if (vet == NULL) return NULL; //OLHAR DEPOIS

    /*percorre tabela preenchendo o vetor*/
    for (int i = 0; i < NTAB; i++){
        for(p = idx->Hash[i]; p != NULL; p = p->next){
            vet[j++] = p;
        }
    }
    return vet;
}

/*funcao de comparacao para colocar vetor em ordem alfabetica*/
static int compara_vet (const void* v1, const void* v2){
    Words **p1 = (Words**)v1;
    Words **p2 = (Words**)v2;

    return strcmp((*p1)->palavra, (*p2)->palavra);
}

int index_print( const Index *idx ){
    int n = idx->num;

    /*cria e ordena o vetor*/
    Words **vet = cria_vetor(idx, n);
    if (vet == NULL) return 20;
    qsort(vet, n, sizeof(Words*), compara_vet);

    /*imprime ocorrências*/
    for(int i = 0; i < n; i++){
        printf("%s = ", vet[i]->palavra);
        Oco* aux = vet[i]->oco;
        while(aux != NULL){
            if(aux->next == NULL) printf("%d", aux->linha);
            else printf("%d, ", aux->linha);
            aux = aux->next;
        }
        printf("\n");
    }

    /*libera vetor*/
    free(vet);
    return 0;
}


//destroy indice
int index_destroy(Index** idx){
    if((*idx) == NULL) return 45;
    Words* aux1, *aux2;
    Oco* aux3, *aux4;
    for (int i = 0; i < NTAB; i++){
        
        aux1 = (*idx)->Hash[i];
        while(aux1 != NULL){//apaga lista de palavras
            aux3 = aux1->oco;
            while(aux3 != NULL){//apaga lista de ocorrencias
                aux4 = aux3->next;
                free(aux3);
                aux3 = aux4;
            }
            aux2 = aux1->next;
            free(aux1);
            aux1 = aux2;
        }
    }
    
    free((*idx)->tf);

    free((*idx));
    (*idx) = NULL;

    //memoria do vetor ocurrences tem que ser liberada na main para liberacao total de memoria

    return 0;
}