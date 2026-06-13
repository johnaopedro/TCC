#!/bin/bash

# Exit on error
set -e

echo "Compilando e executando 001-threshold..."
cd 001-threshold
g++ threshold.cpp -o threshold `pkg-config --cflags --libs opencv4`
echo "Pressione qualquer tecla na janela do OpenCV para continuar para o proximo algoritmo..."
./threshold
cd ..

echo "Compilando e executando 005-equalization..."
cd 005-equalization
g++ equalization.cpp -o equalization `pkg-config --cflags --libs opencv4`
echo "Pressione qualquer tecla na janela do OpenCV para continuar para o proximo algoritmo..."
./equalization
cd ..

echo "Compilando e executando 010-sobel-prewitt-roberts..."
cd 010-sobel-prewitt-roberts
g++ sobel-prewitt-roberts.cpp -o sobel `pkg-config --cflags --libs opencv4`
echo "Pressione qualquer tecla na janela do OpenCV para continuar para o proximo algoritmo..."
./sobel
cd ..

echo "Compilando e executando 013-dilation-erosion..."
cd 013-dilation-erosion
g++ dilation-erosion.cpp -o dilation `pkg-config --cflags --libs opencv4`
echo "Pressione qualquer tecla na janela do OpenCV para continuar para o proximo algoritmo..."
./dilation
cd ..

echo "Compilando e executando 018-linear-interpolation..."
cd 018-linear-interpolation
g++ linear-interpolation.cpp -o interpolation `pkg-config --cflags --libs opencv4`
echo "Pressione qualquer tecla na janela do OpenCV para continuar para o proximo algoritmo..."
./interpolation
cd ..

echo "============================================="
echo "Todos os algoritmos foram executados com sucesso!"
echo "As imagens foram salvas em alta qualidade na pasta tcc-outputs/"
echo "============================================="
