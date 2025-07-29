//AIT FERHAT Thanina
//BENKERROU Lynda

#include "Knn.h"
#include "kmeans.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

// Fonction pour diviser les données en ensembles d'entraînement et de test
void splitData(const std::vector<Image>& allImages, 
               std::vector<Image>& trainingSet, 
               std::vector<Image>& testSet, 
               double trainingRatio = 0.8) {
    
    // Calculer la taille de l'ensemble d'entraînement
    size_t trainingSize = static_cast<size_t>(allImages.size() * trainingRatio);
    
    // Diviser les données
    trainingSet.assign(allImages.begin(), allImages.begin() + trainingSize);
    testSet.assign(allImages.begin() + trainingSize, allImages.end());
    
    std::cout << "Données divisées:" << std::endl;
    std::cout << "  - Ensemble d'entraînement: " << trainingSet.size() << " échantillons" << std::endl;
    std::cout << "  - Ensemble de test: " << testSet.size() << " échantillons" << std::endl;
}

// Fonction pour tester l'algorithme K-NN
void testKNN(const std::vector<Image>& trainingSet, const std::vector<Image>& testSet) {
    std::cout << "\n=== Test de l'algorithme K-NN ===" << std::endl;
    
    // Tester différentes valeurs de k
    std::vector<int> kValues = {1, 3, 5, 7, 9};
    
    for (int k : kValues) {
        std::cout << "\nTest avec k = " << k << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Calculer la matrice de confusion
        auto confusionMatrix = calculateConfusionMatrix(testSet, trainingSet, k);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Calculer les métriques
        double accuracy = calculateAccuracy(confusionMatrix);
        double confusionRate = calculateConfusionRate(confusionMatrix);
        
        std::cout << "  Précision (Accuracy): " << accuracy * 100.0 << "%" << std::endl;
        std::cout << "  Taux de confusion: " << confusionRate * 100.0 << "%" << std::endl;
        std::cout << "  Temps d'exécution: " << duration.count() << " ms" << std::endl;
        
        // Afficher quelques détails de la matrice de confusion
        std::cout << "  Éléments de la matrice de confusion:" << std::endl;
        for (const auto& entry : confusionMatrix) {
            std::cout << "    " << entry.first.first << " -> " << entry.first.second 
                     << ": " << entry.second << std::endl;
        }
    }
}

// Fonction pour tester l'algorithme K-Means
void testKMeans(const std::vector<Image>& allImages) {
    std::cout << "\n=== Test de l'algorithme K-Means ===" << std::endl;
    
    // Tester différents nombres de clusters
    std::vector<int> kValues = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    std::cout << "Analyse des clusters optimaux:" << std::endl;
    
    for (int k : kValues) {
        std::cout << "\nTest avec k = " << k << " clusters" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Créer et entraîner le modèle K-Means
        KMeans kmeans(k);
        kmeans.fit(allImages);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Calculer les métriques
        double silhouetteScore = kmeans.calculateSilhouetteScore(allImages);
        double inertie = kmeans.calculateInertie(allImages);
        
        std::cout << "  Score de silhouette: " << silhouetteScore << std::endl;
        std::cout << "  Inertie: " << inertie << std::endl;
        std::cout << "  Temps d'exécution: " << duration.count() << " ms" << std::endl;
        
        // Afficher la répartition des classes dans les clusters
        assignClassesToClusters(allImages, kmeans.getAssignments(), k);
    }
}

// Fonction pour comparer les deux algorithmes
void compareAlgorithms(const std::vector<Image>& allImages) {
    std::cout << "\n=== Comparaison des algorithmes ===" << std::endl;
    
    // Diviser les données pour K-NN
    std::vector<Image> trainingSet, testSet;
    splitData(allImages, trainingSet, testSet, 0.8);
    
    // Test K-NN avec k=5
    std::cout << "\nPerformances K-NN (k=5):" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto confusionMatrix = calculateConfusionMatrix(testSet, trainingSet, 5);
    auto end = std::chrono::high_resolution_clock::now();
    auto knnDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double knnAccuracy = calculateAccuracy(confusionMatrix);
    
    // Test K-Means avec k=5
    std::cout << "\nPerformances K-Means (k=5):" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    KMeans kmeans(5);
    kmeans.fit(allImages);
    end = std::chrono::high_resolution_clock::now();
    auto kmeansDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double silhouetteScore = kmeans.calculateSilhouetteScore(allImages);
    
    // Résumé comparatif
    std::cout << "\nRésumé comparatif:" << std::endl;
    std::cout << "K-NN:" << std::endl;
    std::cout << "  - Précision: " << knnAccuracy * 100.0 << "%" << std::endl;
    std::cout << "  - Temps: " << knnDuration.count() << " ms" << std::endl;
    std::cout << "  - Type: Apprentissage supervisé" << std::endl;
    
    std::cout << "K-Means:" << std::endl;
    std::cout << "  - Score de silhouette: " << silhouetteScore << std::endl;
    std::cout << "  - Temps: " << kmeansDuration.count() << " ms" << std::endl;
    std::cout << "  - Type: Apprentissage non supervisé" << std::endl;
}

// Fonction principale
int main() {
    std::cout << "=== Projet de Reconnaissance de Formes ===" << std::endl;
    std::cout << "Comparaison des algorithmes K-NN et K-Means" << std::endl;
    std::cout << "Auteurs: AIT FERHAT Thanina, BENKERROU Lynda" << std::endl;
    
    // Chemins vers les données (à adapter selon votre structure)
    std::vector<std::string> chemins_dossiers = {
        "./data/GFD",  // Chemin par défaut pour les données
        // Ajouter d'autres dossiers de données ici
    };
    
    // Vérifier si des données existent
    bool dataFound = false;
    for (const std::string& repertoire : chemins_dossiers) {
        if (fs::exists(repertoire)) {
            dataFound = true;
            std::cout << "\nChargement des données depuis: " << repertoire << std::endl;
            
            // Charger les images
            auto allImages = chargeImages(repertoire);
            
            if (allImages.empty()) {
                std::cout << "Aucune donnée trouvée dans " << repertoire << std::endl;
                continue;
            }
            
            std::cout << "Nombre total d'images chargées: " << allImages.size() << std::endl;
            
            // Diviser les données pour les tests
            std::vector<Image> trainingSet, testSet;
            splitData(allImages, trainingSet, testSet);
            
            // Tester K-NN
            testKNN(trainingSet, testSet);
            
            // Tester K-Means
            testKMeans(allImages);
            
            // Comparer les algorithmes
            compareAlgorithms(allImages);
        }
    }
    
    if (!dataFound) {
        std::cout << "\nAucun dossier de données trouvé!" << std::endl;
        std::cout << "Pour tester les algorithmes, veuillez:" << std::endl;
        std::cout << "1. Créer un dossier 'data/GFD' dans le répertoire du projet" << std::endl;
        std::cout << "2. Y placer vos fichiers de données vectorielles" << std::endl;
        std::cout << "3. Ou modifier les chemins dans le code source" << std::endl;
        
        // Démonstration avec des données exemple
        std::cout << "\nDémonstration avec des données exemple simples:" << std::endl;
        
        // Créer quelques images exemple
        std::vector<Image> exempleImages = {
            Image("A", 1, {1.0, 2.0, 3.0}, "exemple"),
            Image("A", 2, {1.1, 2.1, 3.1}, "exemple"),
            Image("B", 1, {4.0, 5.0, 6.0}, "exemple"),
            Image("B", 2, {4.1, 5.1, 6.1}, "exemple"),
            Image("C", 1, {7.0, 8.0, 9.0}, "exemple"),
            Image("C", 2, {7.1, 8.1, 9.1}, "exemple"),
        };
        
        std::cout << "Test avec " << exempleImages.size() << " échantillons exemple" << std::endl;
        
        // Test rapide K-NN
        std::vector<Image> trainExample(exempleImages.begin(), exempleImages.begin() + 4);
        std::vector<Image> testExample(exempleImages.begin() + 4, exempleImages.end());
        
        auto confusionMatrix = calculateConfusionMatrix(testExample, trainExample, 1);
        double accuracy = calculateAccuracy(confusionMatrix);
        std::cout << "K-NN (k=1) précision: " << accuracy * 100.0 << "%" << std::endl;
        
        // Test rapide K-Means
        KMeans kmeans(3);
        kmeans.fit(exempleImages);
        double silhouette = kmeans.calculateSilhouetteScore(exempleImages);
        std::cout << "K-Means (k=3) silhouette: " << silhouette << std::endl;
    }
    
    std::cout << "\nFin du programme." << std::endl;
    return 0;
}