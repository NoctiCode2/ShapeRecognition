#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>

// Utilisation de la classe Image définie dans Knn.h
#include "Knn.h"

// Classe implémentant l'algorithme KMeans
class KMeans {
public:
    // Constructeur prenant le nombre de clusters
    KMeans(int k);
    
    // Méthodes publiques
    const std::vector<int>& getAssignments() const;
    int getK() const;
    
    // Assigner une image à un cluster en trouvant le centroïde le plus proche
    int assignCluster(const Image& image);
    
    // Calculer le score de silhouette pour évaluer la qualité du clustering
    double calculateSilhouetteScore(const std::vector<Image>& images);
    
    // Calculer l'inertie du clustering
    double calculateInertie(const std::vector<Image>& images);
    
    // Méthode d'entraînement principale
    void fit(const std::vector<Image>& images);

private:
    int k;                                          // Nombre de clusters
    std::vector<std::vector<double>> centroids;     // Centroïdes des clusters
    std::vector<int> assignments;                   // Assignations des points aux clusters

    // Méthodes privées
    void initCentroids(const std::vector<Image>& images);
    bool assignClusters(const std::vector<Image>& images, std::vector<int>& assignments);
    void recalculateCentroids(const std::vector<Image>& images, const std::vector<int>& assignments);
    int findClosestCentroid(const std::vector<double>& values);
    static double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b);
};

// Fonctions utilitaires pour le traitement des données
std::vector<Image> chargeImages(const std::string& repertoire);
void assignClassesToClusters(const std::vector<Image>& images, const std::vector<int>& clusterAssignments, int k);

#endif // KMEANS_H