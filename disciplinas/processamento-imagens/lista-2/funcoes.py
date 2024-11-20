import numpy as np
import matplotlib.pyplot as plt

class PCAAnalysis:
    def __init__(self, data, labels=None, n_components=None):
        """
        Inicializa a classe com o conjunto de dados e o número de componentes desejados.
        
        Parâmetros:
            data (numpy.ndarray): Conjunto de dados com as observações nas linhas e variáveis nas colunas.
            n_components (int, opcional): Número de componentes principais. Se None, mantém todos.
        """
        self.data = data
        self.n_components = n_components
        self.labels = labels
        self.mean = None
        self.data_mean = None
        self.pca_base = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.transformed_data = None

    def data_label(self, labels):
        self.labels = labels

    def def_ncomponents(self, n_components):
        self.n_components = n_components
        
    def truncate(self):
        """
        Reduz o número de componentes principais mantendo os dados truncados para n componentes.
        
        Parâmetros:
            n_components (int): Número de componentes a manter.
        
        Retorna:
            numpy.ndarray: Dados transformados com a quantidade de componentes truncada.
        """
        if self.eigenvectors is None:
             raise ValueError("O PCA ainda não foi ajustado. Execute o método 'fit()' antes.")
        
        if self.n_components is not None:
            matriz = np.zeros((len(self.data), len(self.data)))
            np.fill_diagonal(matriz[:self.n_components, :self.n_components], 1)
            P_PCA = np.dot(self.eigenvectors, matriz)
            self.pca_base = P_PCA
        
        self.transformed_data = np.dot(self.data_mean, self.pca_base)

        return self.transformed_data    

    def fit(self):
        """Realiza o ajuste PCA no conjunto de dados e transforma os dados para a nova base."""

        self.mean = np.mean(self.data, axis=0)
        self.data_mean = self.data - self.mean
        X = self.data_mean
        X_T = np.transpose(X)

        V = (1/len(self.data))*np.dot(X, X_T)

        autovalores, autovetoresV = np.linalg.eigh(V)
        autovetores = np.dot(X_T, autovetoresV)
        autovetores = autovetores / np.linalg.norm(autovetores, axis=0)
        ordem_decrescente = np.argsort(autovalores)[::-1]
        autovalores = autovalores[ordem_decrescente]
        autovetores = autovetores[:, ordem_decrescente]
        
        self.eigenvalues = autovalores
        self.eigenvectors = autovetores
        self.pca_base = autovetores
        self.truncate()

        return self.transformed_data
  
    def plot_primeira_component(self, coeficiente, comp, larg, num_images=3):
        """Plota resultado da media dos dados usando apenas uma dimensão do espaço"""

        coeficientes = coeficiente*[-1, 0, 1]

        fig, axes = plt.subplots(1, 3, figsize=(12, 5))

        for i in range(num_images):
            imagem = self.mean + coeficientes[i] * np.sqrt(self.eigenvalues[0]) * self.eigenvectors[:, 0]
            axes[i].imshow(imagem.reshape(comp, larg), cmap='gray')
            axes[i].set_title(f'Coeficiente {i+1}')
            axes[i].axis('off')  

        plt.show()
    
    def plot_explained_variance(self):
        """Plota o espectro de variância explicada por cada componente principal."""

        if self.eigenvalues is None:
            raise ValueError("O PCA ainda não foi ajustado. Execute o método 'fit()' antes.")

        variancia_explicada = self.eigenvalues / np.sum(self.eigenvalues)
        variancia_explicada_acumulada = np.cumsum(variancia_explicada)

        self.n_components = np.argmax((variancia_explicada_acumulada) >= 0.95) + 1

        print(f"Número de componentes que representam 95% da variância dos dados: {self.n_components}")

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(variancia_explicada)), variancia_explicada, alpha=0.6, label='Variância Explicada Individual')
        plt.step(range(len(variancia_explicada_acumulada)), variancia_explicada_acumulada, where='mid', color='red', label='Variância Explicada Acumulada')
        plt.xlabel('Número de Componentes Principais')
        plt.ylabel('Proporção da Variância Explicada')
        plt.title('Gráfico de Energia dos Autovalores (PCA)')
        plt.legend(loc='best')
        plt.show()


    def plot_pca_projection(self, dim1=0, dim2=1):
        """
        Plota os dados na base PCA em duas dimensões específicas.
        
        Parâmetros:
            dim1 (int): Índice da primeira dimensão a ser plotada.
            dim2 (int): Índice da segunda dimensão a ser plotada.
        """
        if self.transformed_data is None:
            raise ValueError("O PCA ainda não foi ajustado. Execute o método 'fit()' antes.")
        
        if self.labels is not None:
            colors=['blue','red', 'green', 'purple']
            legend =['Sério', 'Sorrindo']

            classes = np.unique(self.labels)

            plt.figure(figsize=(8, 6))

            for i, clas in enumerate(classes):
                
                class_0 = self.transformed_data[self.labels == clas]
                if clas == 0:
                    plt.scatter(class_0[:, dim1], class_0[:, dim2], color=colors[i], label=legend[0])
                else:
                    plt.scatter(class_0[:, dim1], class_0[:, dim2], color=colors[i], label=legend[1])
            
            plt.legend()
                
        else:    
            plt.scatter(self.transformed_data[:, dim1], self.transformed_data[:, dim2], alpha=0.7)
        
        plt.xlabel(f"Componente Principal {dim1 + 1}")
        plt.ylabel(f"Componente Principal {dim2 + 1}")
        plt.title("Projeção dos Dados na Base PCA")
        plt.grid(True)
        plt.show()

    
    def reconstrucion_images(self, data = None, data_transf = None):
        """Visualiza imagens no espaço original após truncamento"""

        if data_transf is None or data is None:
            data = self.data
            data_transf = self.transformed_data
            data_mean = self.mean
        else:
            data_mean = self.mean

        data_reconstructed = data_mean + np.dot(data_transf, np.transpose(self.pca_base))

        num_imagens = 4  

        fig, axes = plt.subplots(2, 4, figsize=(10, 8))  
        fig.subplots_adjust(hspace=0.3)  

        for i in range(num_imagens):
            
            axes[0, i].imshow(data_reconstructed[20*i].reshape(360, 260), cmap='gray')
            axes[0, i].set_title(f'Reconstruída {i+1}')
            axes[0, i].axis('off')  

            
            axes[1, i].imshow(data[20*i].reshape(360, 260), cmap='gray')
            axes[1, i].set_title(f'Original {i+1}')
            axes[1, i].axis('off')

        plt.show()
        return data_reconstructed

    def image_generator(self, n_principal_components, n_imagens):
    
        fig, axs = plt.subplots(1, n_imagens, figsize=(5 * n_imagens, 5))

        if n_imagens == 1:
            axs = [axs]

        for j in range(n_imagens):
        
            img = np.copy(self.mean)
        
            
            for i in range(n_principal_components):
                coeficiente = np.random.uniform(-1, 1)
                img += coeficiente * np.sqrt(self.eigenvalues[i]) * self.eigenvectors[:, i]

            im = axs[j].imshow(img.reshape(360, 260), cmap='gray')
            fig.colorbar(im, ax=axs[j], orientation='vertical')  
            axs[j].grid(False)

        plt.show()

    def transform(self, data):

        if self.pca_base is None:
            raise ValueError("O PCA ainda não foi ajustado. Execute o método 'fit()' antes.")            

        data_mean = data - self.mean

        data_transformed = np.dot(data_mean, self.pca_base)

        return data_transformed 
