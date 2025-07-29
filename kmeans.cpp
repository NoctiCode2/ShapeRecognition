//AIT FERHAT Thanina
//BENKERROU Lynda
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace fs = std::filesystem;

// Classe représentant une image avec ses caractéristiques.
class Image {
public:
    std::string className; // Nom de la classe de l'image.
    int sampleNumber;// Numéro de l'échantillon.
    std::vector<double> values;// Vecteur de caractéristiques de l'image.
    std::string methodName; // Nom de la méthode utilisée pour traiter l'image.
    
    // Constructeur de la classe Image.
    Image(std::string className, int sampleNumber, const std::vector<double>& values, std::string methodName)
            : className(std::move(className)), sampleNumber(sampleNumber), values(values), methodName(std::move(methodName)) {}
};

// Classe implémentant l'algorithme KMeans.
class KMeans {
private:
    std::vector<std::vector<double>> centroids; // AJOUT: stockage des centroïdes
    
public:
    int k;
    std::vector<int> assignments;
    
    KMeans(int k) : k(k) {} // Constructeur prenant le nombre de clusters.
    
    const std::vector<int>& getAssignments() const { return assignments; } // Retourne les affectations de cluster.
    int getK() const { return k; } // Retourne le nombre de clusters.

    // AJOUT: Calcul de la distance euclidienne
    double euclideanDistance(const std::vector<double>& v1, const std::vector<double>& v2) {
        double sum = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            double diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // AJOUT: Trouve le centroïde le plus proche
    int findClosestCentroid(const std::vector<double>& point) {
        int closest = 0;
        double minDistance = euclideanDistance(point, centroids[0]);
        
        for (int i = 1; i < k; ++i) {
            double distance = euclideanDistance(point, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                closest = i;
            }
        }
        return closest;
    }
    
    // Assigner une image à un cluster en trouvant le centroïde le plus proche.
    int assignCluster(const Image& image) {
        int closest = findClosestCentroid(image.values);
        return closest;
    }
    
    // Calculer le score de silhouette pour évaluer la qualité du clustering.
    double calculateSilhouetteScore(const std::vector<Image>& images) {
        std::vector<double> silhouetteScores(images.size(), 0.0);

        for (size_t i = 0; i < images.size(); ++i) {
            double a = 0.0;
            double b = std::numeric_limits<double>::max();
            int countA = std::count(assignments.begin(), assignments.end(), assignments[i]) - 1;

            // Si countA est supérieur à 0, calculez a
            if (countA > 0) {
                for (size_t j = 0; j < images.size(); ++j) {
                    if (i != j && assignments[i] == assignments[j]) {
                        a += euclideanDistance(images[i].values, images[j].values);
                    }
                }
                a /= countA;
            }

            // Calculez b en s'assurant qu'il y a au moins un élément dans le cluster le plus proche
            for (int clusterIdx = 0; clusterIdx < this->k; ++clusterIdx) {
                if (clusterIdx != assignments[i]) {
                    double tempB = 0.0;
                    int countB = 0;
                    for (size_t j = 0; j < images.size(); ++j) {
                        if (assignments[j] == clusterIdx) {
                            tempB += euclideanDistance(images[i].values, images[j].values);
                            countB++;
                        }
                    }
                    if (countB > 0) {
                        tempB /= countB;
                        b = std::min(b, tempB);
                    }
                }
            }

            // CORRECTION: Calcul du score de silhouette complet
            if (std::max(a, b) > 0) {
                silhouetteScores[i] = (b - a) / std::max(a, b);
            }
        }

        // AJOUT: Retour de la moyenne des scores
        double totalScore = 0.0;
        for (double score : silhouetteScores) {
            totalScore += score;
        }
        return totalScore / images.size();
    }
};

// AJOUT: Fonction main pour tester le code
int main() {
    // Exemple d'utilisation avec des données fictives
    std::vector<Image> images = {
        Image("classe1", 1, {1.0, 1.0}, "test"),
        Image("classe1", 2, {1.1, 0.9}, "test"),
        Image("classe2", 1, {5.0, 5.0}, "test"),
        Image("classe2", 2, {5.1, 4.9}, "test")
    };

    KMeans kmeans(2);
    
    // Pour cet exemple, on assigne manuellement les clusters
    kmeans.assignments = {0, 0, 1, 1};
    
    double silhouetteScore = kmeans.calculateSilhouetteScore(images);
    std::cout << "Score de silhouette : " << silhouetteScore << std::endl;

    return 0;
}
