typedef struct matrix Matrix;

int matrix_create( Matrix** m );
/*lê de stdin os elementos diferentes de zero de uma matriz e monta a es-
trutura especificada acima para listas encadeadas, retornando a matriz
criada em m. A entrada consiste dos valores de m e n (número de linhas
e de colunas da matriz) seguidos de triplas (i, j, valor) para os elementos
diferentes de zero da matriz.*/

int matrix_destroy( Matrix* m );
/*devolve toda a memória alocada para a matriz m para a  ́area de memória
dispon ́ıvel (importante:  ́e obrigat ́orio neste trabalho o uso de alocação
dinâmica de mem ́oria para implementar as representa ̧c ̃oes das matrizes!)*/

int matrix_print( const Matrix* m );
/*imprime a matriz m para stdout no mesmo formato usado por matrix_create().*/

int matrix_add( const Matrix* m, const Matrix* n, Matrix** r );
/*recebe como parˆametros as matrizes m e n, retornando em r a soma das
mesmas (a estrutura da matriz retornada deve ser alocada dinamicamente
pela própria operação).*/

int matrix_multiply( const Matrix* m, const Matrix* n, Matrix** r );
/*recebe como parâmetros as matrizes m e n, retornando em r a multiplicação
das mesmas (a estrutura da matriz retornada deve ser alocada dinamica-
mente pela própria operação).*/

int matrix_transpose( const Matrix* m, Matrix** r );
/*recebe como parametro a matriz m, retornando em r a matriz mT – a
transposta de m (a estrutura da matriz retornada deve ser alocada dina-
micamente pela pr ́opria operação).*/

int matrix_getelem( const Matrix* m, int x, int y, float *elem );
/*retorna o valor do elemento (x, y) da matriz m em elem.*/

int matrix_setelem( Matrix* m, int x, int y, float elem );
/*atribui ao elemento (x, y) da matriz m o valor elem.*/