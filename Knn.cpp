
## 2. 🔧 Knn.cpp - Corrections minimales

```cpp name=Knn.cpp
//AIT FERHAT Thanina

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <random>
#include <cmath>

namespace fs = std::filesystem;

//Récupère le nom de la classe/ num échantillon, vecteur de données, le nom de la méthode
class Image {
public:
    std::string className;
    int sampleNumber;
    std::vector<double> values;
    std::string methodName;

    Image(std::string className, int sampleNumber, const std::vector<double>& values, std::string methodName)
            : className(std::move(className)), sampleNumber(sampleNumber), values(values), methodName(std::move(methodName)) {}
};

//lecture de fichiers d'un dossier et stockage dans des vecteurs
std::vector<double> readVectorsFromFolders(const std::string& folderName) {
    std::ifstream file(folderName);
    std::vector<double> vect;

    if (file) {
        double number;
        while (file >> number) {
            vect.push_back(number);
        }
        file.close();
    } else {
        std::cerr << "Erreur lors de l'ouverture du fichier : " << folderName << std::endl;
    }

    return vect;
}

// fonction pour calculer la distance euclidienne entre deux vecteurs
double distance(const std::vector<double>& v1, const std::vector<double>& v2) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// fonction pour prédire la classe d'une image en utilisant k-NN
std::string predictKNN(const std::vector<Image>& trainingSet, const std::vector<double>& queryVector, int k) {
    // un vecteur pour stocker les distances entre l'image de requête et les images d'entraînement
    std::vector<std::pair<double, std::string>> distances;

    // Calcul distance entre la requête et chaque image d'entraînement
    for (const Image& image : trainingSet) {
        double dist = distance(queryVector, image.values);
        distances.emplace_back(dist, image.className);
    }

    // Tri les distances par ordre croissant
    std::sort(distances.begin(), distances.end());

    // Compte les occurrences de chaque classe parmi les k voisins les plus proches
    std::map<std::string, int> classCounts;
    for (int i = 0; i < k; ++i) {
        classCounts[distances[i].second]++;
    }

    // Trouve la classe la plus fréquente parmi les k voisins les plus proches
    std::string predictedClass;
    int maxCount = 0;
    for (const auto& pair : classCounts) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            predictedClass = pair.first;
        }
    }

    return predictedClass;
}

//fonction pour calculer la matrice de confusion - CORRECTION: Implémentation complète
std::map<std::pair<std::string, std::string>, int> calculateConfusionMatrix(const std::vector<Image>& testSet, const std::vector<Image>& trainingSet, int k) {
    std::map<std::pair<std::string, std::string>, int> confusionMatrix;

    // Parcours de chaque image de test pour prédire sa classe
    for (const Image& testImage : testSet) {
        std::string predictedClass = predictKNN(trainingSet, testImage.values, k);
        confusionMatrix[{testImage.className, predictedClass}]++;
    }

    return confusionMatrix;
}

// AJOUT: Fonction pour afficher la matrice de confusion
void displayConfusionMatrix(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix) {
    std::cout << "Matrice de confusion (Vraie classe, Classe prédite) -> Nombre:" << std::endl;
    for (const auto& entry : confusionMatrix) {
        std::cout << "(" << entry.first.first << ", " << entry.first.second << ") -> " << entry.second << std::endl;
    }
}

// AJOUT: Fonction main pour tester le code
int main() {
    // Exemple d'utilisation avec des données fictives
    std::vector<Image> trainingSet = {
        Image("cercle", 1, {1.0, 1.0, 2.0}, "test"),
        Image("carre", 1, {3.0, 3.0, 4.0}, "test"),
        Image("cercle", 2, {1.1, 0.9, 2.1}, "test")
    };
    
    std::vector<Image> testSet = {
        Image("cercle", 3, {1.05, 0.95, 2.05}, "test"),
        Image("carre", 2, {2.9, 3.1, 3.9}, "test")
    };

    int k = 2;
    auto confusionMatrix = calculateConfusionMatrix(testSet, trainingSet, k);
    displayConfusionMatrix(confusionMatrix);

    return 0;
}
