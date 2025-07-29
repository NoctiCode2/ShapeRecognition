
## 2. 🔧 Knn.cpp corrigé

```cpp name=Knn.cpp
//AIT FERHAT Thanina
//Projet d'évaluation des méthodes classiques de reconnaissance des formes

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <random>
#include <cmath>
#include <iomanip>

namespace fs = std::filesystem;

// Classe représentant une image avec ses caractéristiques
class Image {
public:
    std::string className;
    int sampleNumber;
    std::vector<double> values;
    std::string methodName;

    Image(std::string className, int sampleNumber, const std::vector<double>& values, std::string methodName)
            : className(std::move(className)), sampleNumber(sampleNumber), values(values), methodName(std::move(methodName)) {}
};

// Lecture de fichiers d'un dossier et stockage dans des vecteurs
std::vector<double> readVectorsFromFolders(const std::string& folderName) {
    std::ifstream file(folderName);
    std::vector<double> vect;

    if (!file.is_open()) {
        std::cerr << "Erreur lors de l'ouverture du fichier : " << folderName << std::endl;
        return vect;
    }

    double number;
    while (file >> number) {
        vect.push_back(number);
    }
    
    if (vect.empty()) {
        std::cerr << "Attention : Le fichier " << folderName << " est vide ou ne contient pas de données valides." << std::endl;
    }
    
    file.close();
    return vect;
}

// Fonction pour calculer la distance euclidienne entre deux vecteurs
double distance(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        std::cerr << "Erreur : Les vecteurs n'ont pas la même taille (" << v1.size() << " vs " << v2.size() << ")" << std::endl;
        return std::numeric_limits<double>::max();
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Fonction pour prédire la classe d'une image en utilisant k-NN
std::string predictKNN(const std::vector<Image>& trainingSet, const std::vector<double>& queryVector, int k) {
    if (trainingSet.empty()) {
        std::cerr << "Erreur : Ensemble d'entraînement vide" << std::endl;
        return "";
    }
    
    if (k <= 0 || k > static_cast<int>(trainingSet.size())) {
        std::cerr << "Erreur : k doit être entre 1 et " << trainingSet.size() << std::endl;
        return "";
    }

    // Vecteur pour stocker les distances entre l'image de requête et les images d'entraînement
    std::vector<std::pair<double, std::string>> distances;

    // Calcul distance entre la requête et chaque image d'entraînement
    for (const Image& image : trainingSet) {
        double dist = distance(queryVector, image.values);
        if (dist != std::numeric_limits<double>::max()) {
            distances.emplace_back(dist, image.className);
        }
    }

    if (distances.empty()) {
        std::cerr << "Erreur : Aucune distance valide calculée" << std::endl;
        return "";
    }

    // Tri des distances par ordre croissant
    std::sort(distances.begin(), distances.end());

    // Compte les occurrences de chaque classe parmi les k voisins les plus proches
    std::map<std::string, int> classCounts;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i) {
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

// Fonction pour calculer la matrice de confusion - IMPLÉMENTATION COMPLÈTE
std::map<std::pair<std::string, std::string>, int> calculateConfusionMatrix(
    const std::vector<Image>& testSet, 
    const std::vector<Image>& trainingSet, 
    int k) {
    
    std::map<std::pair<std::string, std::string>, int> confusionMatrix;

    // Parcours de chaque image de test
    for (const Image& testImage : testSet) {
        // Prédiction de la classe avec k-NN
        std::string predictedClass = predictKNN(trainingSet, testImage.values, k);
        
        if (!predictedClass.empty()) {
            // Incrémentation de la cellule correspondante dans la matrice de confusion
            confusionMatrix[{testImage.className, predictedClass}]++;
        }
    }

    return confusionMatrix;
}

// Fonction pour afficher la matrice de confusion
void displayConfusionMatrix(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix) {
    std::cout << "\n=== MATRICE DE CONFUSION ===" << std::endl;
    std::cout << "Format: (Vraie classe, Classe prédite) -> Nombre" << std::endl;
    
    for (const auto& entry : confusionMatrix) {
        std::cout << "(" << entry.first.first << ", " << entry.first.second << ") -> " << entry.second << std::endl;
    }
}

// Fonction pour calculer la précision globale
double calculateAccuracy(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix) {
    int correct = 0;
    int total = 0;
    
    for (const auto& entry : confusionMatrix) {
        total += entry.second;
        if (entry.first.first == entry.first.second) {
            correct += entry.second;
        }
    }
    
    return total > 0 ? static_cast<double>(correct) / total * 100.0 : 0.0;
}

// Fonction pour créer des données d'exemple
std::vector<Image> createSampleData() {
    std::vector<Image> data;
    
    // Exemples pour classe "cercle"
    data.emplace_back("cercle", 1, std::vector<double>{1.0, 1.0, 2.0, 2.0}, "test");
    data.emplace_back("cercle", 2, std::vector<double>{1.1, 0.9, 2.1, 1.9}, "test");
    data.emplace_back("cercle", 3, std::vector<double>{0.9, 1.1, 1.9, 2.1}, "test");
    
    // Exemples pour classe "carré"
    data.emplace_back("carre", 1, std::vector<double>{3.0, 3.0, 4.0, 4.0}, "test");
    data.emplace_back("carre", 2, std::vector<double>{3.1, 2.9, 4.1, 3.9}, "test");
    data.emplace_back("carre", 3, std::vector<double>{2.9, 3.1, 3.9, 4.1}, "test");
    
    // Exemples pour classe "triangle"
    data.emplace_back("triangle", 1, std::vector<double>{5.0, 5.0, 6.0, 6.0}, "test");
    data.emplace_back("triangle", 2, std::vector<double>{5.1, 4.9, 6.1, 5.9}, "test");
    data.emplace_back("triangle", 3, std::vector<double>{4.9, 5.1, 5.9, 6.1}, "test");
    
    return data;
}

// Fonction principale pour tester l'algorithme K-NN
int main() {
    std::cout << "=== TEST DE L'ALGORITHME K-NEAREST NEIGHBORS ===" << std::endl;
    
    // Création de données d'exemple
    std::vector<Image> sampleData = createSampleData();
    
    // Division en ensemble d'entraînement et de test
    std::vector<Image> trainingSet, testSet;
    
    for (size_t i = 0; i < sampleData.size(); ++i) {
        if (i % 3 == 0) {
            testSet.push_back(sampleData[i]);
        } else {
            trainingSet.push_back(sampleData[i]);
        }
    }
    
    std::cout << "Ensemble d'entraînement : " << trainingSet.size() << " échantillons" << std::endl;
    std::cout << "Ensemble de test : " << testSet.size() << " échantillons" << std::endl;
    
    // Test avec différentes valeurs de k
    for (int k = 1; k <= 3; ++k) {
        std::cout << "\n--- Test avec k = " << k << " ---" << std::endl;
        
        // Calcul de la matrice de confusion
        auto confusionMatrix = calculateConfusionMatrix(testSet, trainingSet, k);
        
        // Affichage des résultats
        displayConfusionMatrix(confusionMatrix);
        
        double accuracy = calculateAccuracy(confusionMatrix);
        std::cout << "Précision globale : " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    }
    
    // Test de prédiction individuelle
    std::cout << "\n=== TEST DE PRÉDICTION INDIVIDUELLE ===" << std::endl;
    std::vector<double> testVector = {1.05, 0.95, 2.05, 1.95};
    std::string prediction = predictKNN(trainingSet, testVector, 3);
    
    std::cout << "Vecteur de test : [";
    for (size_t i = 0; i < testVector.size(); ++i) {
        std::cout << testVector[i];
        if (i < testVector.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Classe prédite : " << prediction << std::endl;
    
    return 0;
}
