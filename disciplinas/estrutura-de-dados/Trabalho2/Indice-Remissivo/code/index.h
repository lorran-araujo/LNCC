typedef struct index Index;

int index_createfrom (const char *key_file, const char *text_file, Index **idx);
/*lÊ do arquivo de nome key_file as palavras-chave a serem usadas na
construção do índice remissivo do texto contido no arquivo texto de nome
text_file. O índice criado deve ser retornado em idx. As palavras chave em 
key_file devem estar cada uma em uma linha separada, como ilustrado abaixo:

programs
easy
by
and
be
to*/

int index_get(const Index *idx, const char *key, int **occurrences, int *num_occurrences);
/*devolve em occurrences um vetor de inteiros (alocado dinamicamente
pela operação) contendo todas as ocorrências (linhas) da palavra-chave
key armazenadas pelo índice remissivo apontado por idx. O número de
entradas de occurrences é devolvido em num_occurrences.*/

int index_put(Index *idx, const char *key);
/*inclui no índice remissivo a palavra-chave key, associando-a a todas as
ocorrÊncias presentes no arquivo texto usado na construção do índice.
Caso a palavra-chave já exista no índice remissivo, as ocorrências da
mesma no arquivo texto devem ser atualizadas (´util quando o arquivo
texto ´e alterado durante a execução de um programa que usa o TAD).*/

int index_print( const Index *idx );
/*imprime para stdout o índice remissivo completo, em ordem alfabética,
referenciado por idx. O formato de saída desse índice deve ser como
ilustrado abaixo

and: 4, 5, 6
be: 3
...*/

int index_destroy(Index** idx);