#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <vector>
#include <map>
#include <string>

// Classe représentant une image avec ses caractéristiques
class Image {
public:
    std::string className;      // Nom de la classe de l'image
    int sampleNumber;          // Numéro de l'échantillon
    std::vector<double> values; // Vecteur de caractéristiques de l'image
    std::string methodName;     // Nom de la méthode utilisée pour traiter l'image

    // Constructeur de la classe Image
    Image(std::string className, int sampleNumber, const std::vector<double>& values, std::string methodName);
};

// Fonctions de lecture et traitement des données
std::vector<double> readVectorsFromFolders(const std::string& folderName);

// Fonction de calcul de distance euclidienne
double distance(const std::vector<double>& v1, const std::vector<double>& v2);

// Algorithme KNN
std::string predictKNN(const std::vector<Image>& trainingSet, const std::vector<double>& testVector, int k);

// Fonctions de calcul de la matrice de confusion et métriques
std::map<std::pair<std::string, std::string>, int> calculateConfusionMatrix(
    const std::vector<Image>& testSet, 
    const std::vector<Image>& trainingSet, 
    int k
);

// Fonctions de calcul des métriques de performance
double calculateAccuracy(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix);
double calculateConfusionRate(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix);
std::map<std::string, double> calculateRecall(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix);
std::map<std::string, double> calculatePrecision(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix);
std::map<std::string, double> calculateFMeasure(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix);

#endif // KNN_H