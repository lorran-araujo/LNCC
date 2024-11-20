import streamlit as st
from PIL import Image
import numpy as np
from scipy.signal import convolve

# Funções de filtro
def apply_gaussian_filter(image_data):
    size = 21
    sigma = 3
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    gauss_filter = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gauss_filter /= np.sum(gauss_filter)
    return convolve(image_data, gauss_filter)

def apply_derivative_x(image_data):
    size = 21
    sigma = 3
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    dx_filter = - (x / (np.power(sigma, 3) * np.sqrt(2 * np.pi))) * np.exp(- (x**2 + y**2) / (2 * (sigma ** 2)))
    dx_filter /= np.sum(np.abs(dx_filter))
    return convolve(image_data, dx_filter)

def apply_derivative_y(image_data):
    size = 21
    sigma = 3
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    dy_filter = - (y / (np.power(sigma, 3) * np.sqrt(2 * np.pi))) * np.exp(- (x**2 + y**2) / (2 * (sigma ** 2)))
    dy_filter /= np.sum(np.abs(dy_filter))
    return convolve(image_data, dy_filter)

# Normalizar a imagem para o intervalo [0, 1]
def normalize_image(image_data):
    return np.clip(image_data, 0, 1)



# Interface do Streamlit
st.title("Aplicador de Filtros de Imagem")

uploaded_file = st.file_uploader("Carregar uma imagem", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image_data = np.array(image)

    st.image(image, caption='Imagem Original', use_container_width=True)

    filter_type = st.selectbox("Escolha o filtro", ["Gaussiano", "Derivada em x", "Derivada em y"])

    if st.button("Aplicar Filtro"):
        if filter_type == "Gaussiano":
            filtered_image = apply_gaussian_filter(image_data)
        elif filter_type == "Derivada em x":
            filtered_image = apply_derivative_x(image_data)
        elif filter_type == "Derivada em y":
            filtered_image = apply_derivative_y(image_data)

        # Dentro do código de aplicação do filtro
        filtered_image_normalized = normalize_image(filtered_image) 

        # Exibir a imagem normalizada
        st.image(filtered_image_normalized, caption=f'Imagem com filtro {filter_type}', use_container_width=True)