# Makefile pour le projet ShapeRecognition

# Compilateur et options
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -I.

# Fichiers sources
KNN_SOURCES = Knn.cpp
KMEANS_SOURCES = kmeans.cpp
MAIN_SOURCES = main.cpp

# Fichiers objets
KNN_OBJECTS = $(KNN_SOURCES:.cpp=.o)
KMEANS_OBJECTS = $(KMEANS_SOURCES:.cpp=.o)
MAIN_OBJECTS = $(MAIN_SOURCES:.cpp=.o)

# Exécutables
KNN_TARGET = knn
KMEANS_TARGET = kmeans
MAIN_TARGET = shape_recognition

# Règle par défaut
all: $(KNN_TARGET) $(KMEANS_TARGET) $(MAIN_TARGET)

# Compilation de l'exécutable KNN
$(KNN_TARGET): $(KNN_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilation de l'exécutable KMeans
$(KMEANS_TARGET): $(KMEANS_OBJECTS) knn_lib.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilation de l'exécutable principal
$(MAIN_TARGET): $(MAIN_OBJECTS) knn_lib.o kmeans_lib.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilation des fichiers objets
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compilation spéciale pour knn en tant que bibliothèque (sans main)
knn_lib.o: Knn.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DKNN_LIB -c $< -o $@

# Compilation spéciale pour kmeans en tant que bibliothèque (sans main)
kmeans_lib.o: kmeans.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DKMEANS_LIB -c $< -o $@

# Nettoyage
clean:
	rm -f *.o $(KNN_TARGET) $(KMEANS_TARGET) $(MAIN_TARGET) test_*

# Nettoyage complet
distclean: clean
	rm -f *~ *.bak

# Installation (optionnelle)
install: all
	mkdir -p bin
	cp $(KNN_TARGET) $(KMEANS_TARGET) $(MAIN_TARGET) bin/

# Tests
test: $(KNN_TARGET) $(KMEANS_TARGET)
	@echo "Tests nécessitent des données d'exemple"
	@echo "Exécutez ./$(KNN_TARGET) ou ./$(KMEANS_TARGET) avec vos données"

# Aide
help:
	@echo "Cibles disponibles:"
	@echo "  all       - Compile tous les exécutables"
	@echo "  knn       - Compile l'algorithme K-NN"
	@echo "  kmeans    - Compile l'algorithme K-Means"
	@echo "  main      - Compile l'exécutable principal"
	@echo "  clean     - Supprime les fichiers objets et exécutables"
	@echo "  distclean - Nettoyage complet"
	@echo "  install   - Installe les exécutables dans le dossier bin/"
	@echo "  test      - Exécute les tests"
	@echo "  help      - Affiche cette aide"

.PHONY: all clean distclean install test help