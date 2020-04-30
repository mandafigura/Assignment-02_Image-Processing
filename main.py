# # # # # # # # # # #  HEADER  # # # # # # # # # # #
#                                                  #
#        ALUNA: Amanda Carrijo Viana Figur         #
#                N. USP: 8937736                   #
#        ALUNO: Luiz Augusto Vieira Manoel         #
#                N. USP: 8937308                   #
#   CURSO: Mestrado em Ciências de Computação e    #
#        Matemática Computacional (PPG-CCMC)       #
#            ANO DE INGRESSO: 2020/2019            #
#             ASSIGNMENT 2 - FILTERING             #
#                                                  #
##  # # # # # # # # # # # # # # # # # # # # # # # ##

import sys                          #stdin, stdout & stderr
import numpy as np                  #mexe direitinho com números
import imageio as img               #mexe direitinho com as imagens
import matplotlib.pyplot as plt     #plota as coisinhas

#Função que plota duas imagens lado a lado. r = referencia, m = modificada
def plot_compare(r, m):
    plt.figure(figsize=(12, 12))

    # defines a panel to show the images side by side
    plt.subplot(121)  # panel with 1 row, 2 columns, to show the image at the first (1st) position
    plt.imshow(r, cmap="gray")
    plt.axis('off')   # remove axis with numbers

    plt.subplot(122)  # panel with 1 row, 2 columns, to show the image at the second (2nd) position
    plt.imshow(m, cmap="gray")
    plt.axis('off')

    plt.show()

#normaliza uma imagem referência r
def image_normalization(r):
    a = np.amin(r)   #menor intensidade da imagem como float pra calcular valores
    b = np.amax(r)   #maior intensidade da imagem como float pra calcular valores
    #normaliza intensidades da imagem
    r = ((r - a)*255)/b
    return r

#aplica a convolução w a uma imagem r=referencia e retorna a matriz modificada g como float32
#OBS: APLICAR O PADDING NECESSÁRIO ANTES
def image_convolution(r, w):
    #calcula o tamanha da matriz w de convolução
    n, m = w.shape

    #Encontra o centro da matriz de convolução: o tamanho das linhas e colunas de w é sempre ímpar
    a = int((n - 1) / 2) #linha
    b = int((m - 1) / 2) #coluna

    N, M = r.shape

    #flipped filter pra aplicar a convolução
    w_flip = np.flip(np.flip(w, 0), 1)

    #nova imagem para salvar os valores com a convolução aplicada
    g = np.zeros(r.shape, dtype=np.float32)

    #calculando o valor para cada pixel da imagem, ignorando os pixels das bordas (que aplicamos padding)
    for x in range(a, N - a):
        for y in range(b, M - b):
            # gets subimage
            sub_r = r[x - a: x + a + 1, y - b:y + b + 1]
            # computes g at (x,y)
            g[x, y] = np.sum(np.multiply(sub_r, w_flip))
    return g


#Método 1
#Função que aplica o filtro bilateral. Em ordem: Imagem de Referência, tamanho do filtro n
#----------------------------------------------- parâmetro sigma_s e parâmetro sigma_r
def bilateral_filter(r, n, ss, sr):
    pass

#Método 2
#Função que aplica o filtro laplaciano e retorna a imagem modificada. Em ordem: Imagem de Referência, parâmetro c e kernel
def unsharp_mask(r, c, k):
    #determinando a matriz que utilizaremos para fazer a convolução
    if k==1:
        w = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    elif k==2:
        w = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

    #converte matrizes pra precisão de ponto flutuante antes de todas as contas
    r = r.astype(np.float32)
    w = w.astype(np.float32)
    #convolucionando a imagem
    m = image_convolution(r,w)
    #normalizando a imagem modificada
    m = image_normalization(m)
    #aplicando a operação com o parâmetro c na imagem normalizada
    m = m*c + r
    #normalizando a imagem após o parâmetro c
    m = image_normalization(m)

    return (m.astype(np.uint8))


#Método 3
#função que sei lá. Em ordem: Imagem de Referência, parâmetro sigma_col e parâmetro sigma_row
def vignette_filter(r, row, col):
    pass

#Função que prepara a imagem r=referencia pra aplicar a convulução com um padding de tamanho determinado de linhas e colunas
def image_padding(r,prow,pcol):
    r = np.pad(r, ((prow,prow),(pcol,pcol)), 'constant') #(array, tamanho_do_padding, valor_do_padding_defaut_eh_zero)
    #print(r)
    return r


#função que calcula o erro, recebe (respectivamente) a imagem de referência e a imagem modificada
def calculo_erro(r, m):
    #converte informações das intensidades para float (evitar erros nos cálculos)
    m = m.astype(np.float32)
    r = r.astype(np.float32)

    #cria uma matriz auxiliar na qual executaremos as contas necessárias
    aux = m - r             #calcula a diferença entre cada termo da matriz modificada e da matriz referencia
    aux = np.power(aux, 2)  #eleva cada termo da matriz ao quadrado
    soma = aux.sum()        #soma todos os termos da matriz
    RSE = np.sqrt(soma)     #a raiz quadrada da soma resulta no erro desejado

    #imprime o número RSE com 4 casas decimais de precisão e ajusta a identação
    print (f"{RSE:.4f}", sep='', end='')


#função que salva a imagem caso a condição indique que sim
def save_img(img, name, condition):
    if condition == 1:
        img.imwrite(name, img)


# Reading inputs
filename = str(input()).rstrip()
method = int(input())
save = int(input())

input_img = img.imread(filename)

if method == 1:
    #recebe parametros
    filter_size = int(input())
    sigma_s = float(input())
    sigma_r = float(input())

    #aplica o filtro
    output_img = bilateral_filter(input_img,filter_size,sigma_s,sigma_r)



elif method == 2:
    #recebe parametros
    par_c = float(input())
    kernel = int(input())
    #faz o padding da imagem
    input_img = image_padding(input_img, 1, 1)
    #aplica o filtro
    output_img = unsharp_mask(input_img,par_c,kernel)


elif method == 3:
    #recebe parametros
    sigma_row = float(input())
    sigma_col = float(input())

    #aplica o filtro
    output_img = vignette_filter(input_img,sigma_row,sigma_col)


calculo_erro(input_img,output_img)                  #computa o erro RSE
plot_compare(input_img,output_img)                  #compara imagens lado a lado
save_img(output_img,filename + "_nova.png", save)   #salva a imagem modificada, caso indicado