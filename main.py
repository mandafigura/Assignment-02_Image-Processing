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

#Método 1
#Função que aplica o filtro bilateral. Em ordem: Imagem de Referência, tamanho do filtro n
#----------------------------------------------- parâmetro sigma_s e parâmetro sigma_r
#def bilateral_filter(r, n, ss, sr):

#Método 2
#Função que aplica o filtro laplaciano. Em ordem: Imagem de Referência, parâmetro c e kernel
#def unsharp_mask(r, c, k):


#Método 3
#função que sei lá. Em ordem: Imagem de Referência, parâmetro sigma_col e parâmetro sigma_row
#def vignette_filter(r, row, col):


#Função que prepara a imagem pra convulução
def image_padding(r):
    r = np.pad(r, 1, 'constant')
    print(r)
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
input_img = image_padding(input_img)

if method == 1:
    filter_size = int(input())
    sigma_s = float(input())
    sigma_r = float(input())

    output_img = bilateral_filter(input_img,filter_size,sigma_s,sigma_r)



elif method == 2:
    par_c = float(input())
    kernel = int(input())

    output_img = unsharp_mask(input_img,par_c,kernel)



elif method == 3:
    sigma_row = float(input())
    sigma_col = float(input())

    output_img = (input_img,sigma_row,sigma_col)



calculo_erro(input_img,output_img)
plot_compare(input_img,output_img)
save_img(output_img,filename + "_nova.png", save)

