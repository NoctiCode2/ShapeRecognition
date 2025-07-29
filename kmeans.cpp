//AIT FERHAT Thanina
//BENKERROU Lynda
//Projet d'évaluation des méthodes classiques de reconnaissance des formes

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

// Classe implémentant l'algorithme KMeans - IMPLÉMENTATION COMPLÈTE
class KMeans {
private:
    int k;
    std::vector<std::vector<double>> centroids;
    std::vector<int> assignments;
    int maxIterations;
    double tolerance;

public:
    KMeans(int k, int maxIter = 100, double tol = 1e-4) 
        : k(k), maxIterations(maxIter), tolerance(tol) {}

    const std::vector<int>& getAssignments() const { 
        return assignments; 
    }

    const std::vector<std::vector<double>>& getCentroids() const {
        return centroids;
    }

    int getK() const { 
        return k; 
    }

    // Calcul de la distance euclidienne entre deux vecteurs
    double euclideanDistance(const std::vector<double>& v1, const std::vector<double>& v2) const {
        if (v1.size() != v2.size()) {
            return std::numeric_limits<double>::max();
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            double diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // Trouve le centroïde le plus proche d'un point
    int findClosestCentroid(const std::vector<double>& point) const {
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

    // Assigne une image à un cluster
    int assignCluster(const Image& image) {
        return findClosestCentroid(image.values);
    }

    // Met à jour les centroïdes
    void updateCentroids(const std::vector<Image>& images) {
        std::vector<std::vector<double>> newCentroids(k, std::vector<double>(centroids[0].size(), 0.0));
        std::vector<int> counts(k, 0);

        // Somme des points pour chaque cluster
        for (size_t i = 0; i < images.size(); ++i) {
            int cluster = assignments[i];
            for (size_t j = 0; j < images[i].values.size(); ++j) {
                newCentroids[cluster][j] += images[i].values[j];
            }
            counts[cluster]++;
        }

        // Calcul de la moyenne pour chaque cluster
        for (int i = 0; i < k; ++i) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < newCentroids[i].size(); ++j) {
                    newCentroids[i][j] /= counts[i];
                }
            }
        }

        centroids = newCentroids;
    }

    // Initialise les centroïdes aléatoirement
    void initializeCentroids(const std::vector<Image>& images) {
        if (images.empty()) return;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, images.size() - 1);

        centroids.clear();
        centroids.resize(k);

        // Sélection aléatoire de k points comme centroïdes initiaux
        for (int i = 0; i < k; ++i) {
            int randomIndex = dis(gen);
            centroids[i] = images[randomIndex].values;
        }
    }

    // Entraînement du modèle K-Means
    bool fit(const std::vector<Image>& images) {
        if (images.empty() || k <= 0 || k > static_cast<int>(images.size())) {
            std::cerr << "Erreur : Paramètres invalides pour K-Means" << std::endl;
            return false;
        }

        // Initialisation
        initializeCentroids(images);
        assignments.resize(images.size());

        std::cout << "Démarrage de l'algorithme K-Means avec k=" << k << std::endl;

        for (int iter = 0; iter < maxIterations; ++iter) {
            std::vector<std::vector<double>> oldCentroids = centroids;

            // Assignation des points aux clusters
            for (size_t i = 0; i < images.size(); ++i) {
                assignments[i] = findClosestCentroid(images[i].values);
            }

            // Mise à jour des centroïdes
            updateCentroids(images);

            // Vérification de la convergence
            double maxChange = 0.0;
            for (int i = 0; i < k; ++i) {
                double change = euclideanDistance(oldCentroids[i], centroids[i]);
                maxChange = std::max(maxChange, change);
            }

            std::cout << "Itération " << iter + 1 << " - Changement max: " << maxChange << std::endl;

            if (maxChange < tolerance) {
                std::cout << "Convergence atteinte après " << iter + 1 << " itérations" << std::endl;
                return true;
            }
        }

        std::cout << "Nombre maximum d'itérations atteint" << std::endl;
        return true;
    }

    // Calcul du score de silhouette pour évaluer la qualité du clustering
    double calculateSilhouetteScore(const std::vector<Image>& images) {
        if (images.size() <= 1) return 0.0;
        
        std::vector<double> silhouetteScores(images.size(), 0.0);

        for (size_t i = 0; i < images.size(); ++i) {
            double a = 0.0; // Distance moyenne intra-cluster
            double b = std::numeric_limits<double>::max(); // Distance moyenne inter-cluster la plus petite
            
            // Calcul de a (cohésion intra-cluster)
            int countA = 0;
            for (size_t j = 0; j < images.size(); ++j) {
                if (i != j && assignments[i] == assignments[j]) {
                    a += euclideanDistance(images[i].values, images[j].values);
                    countA++;
                }
            }
            if (countA > 0) {
                a /= countA;
            }

            // Calcul de b (séparation inter-cluster)
            for (int cluster = 0; cluster < k; ++cluster) {
                if (cluster != assignments[i]) {
                    double tempB = 0.0;
                    int countB = 0;
                    
                    for (size_t j = 0; j < images.size(); ++j) {
                        if (assignments[j] == cluster) {
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

            // Calcul du score de silhouette pour ce point
            if (std::max(a, b) > 0) {
                silhouetteScores[i] = (b - a) / std::max(a, b);
            }
        }

        // Moyenne des scores de silhouette
        double totalScore = 0.0;
        for (double score : silhouetteScores) {
            totalScore += score;
        }
        
        return totalScore / images.size();
    }

    // Affiche les informations sur les clusters
    void displayClusterInfo(const std::vector<Image>& images) {
        std::cout << "\n=== INFORMATIONS SUR LES CLUSTERS ===" << std::endl;
        
        // Compte des éléments par cluster
        std::vector<int> clusterCounts(k, 0);
        std::vector<std::map<std::string, int>> classDistribution(k);
        
        for (size_t i = 0; i < images.size(); ++i) {
            int cluster = assignments[i];
            clusterCounts[cluster]++;
            classDistribution[cluster][images[i].className]++;
        }
        
        for (int i = 0; i < k; ++i) {
            std::cout << "Cluster " << i << " : " << clusterCounts[i] << " éléments" << std::endl;
            std::cout << "  Distribution des classes:" << std::endl;
            for (const auto& pair : classDistribution[i]) {
                std::cout << "    " << pair.first << " : " << pair.second << std::endl;
            }
            std::cout << "  Centroïde: [";
            for (size_t j = 0; j < centroids[i].size(); ++j) {
                std::cout << std::fixed << std::setprecision(3) << centroids[i][j];
                if (j < centroids[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl << std::endl;
        }
    }
};

// Fonction pour créer des données d'exemple
std::vector<Image> createSampleData() {
    std::vector<Image> data;
    
    // Cluster 1 - Cercles
    data.emplace_back("cercle", 1, std::vector<double>{1.0, 1.0, 2.0, 2.0}, "test");
    data.emplace_back("cercle", 2, std::vector<double>{1.1, 0.9, 2.1, 1.9}, "test");
    data.emplace_back("cercle", 3, std::vector<double>{0.9, 1.1, 1.9, 2.1}, "test");
    data.emplace_back("cercle", 4, std::vector<double>{1.2, 0.8, 2.2, 1.8}, "test");
    
    // Cluster 2 - Carrés
    data.emplace_back("carre", 1, std::vector<double>{5.0, 5.0, 6.0, 6.0}, "test");
    data.emplace_back("carre", 2, std::vector<double>{5.1, 4.9, 6.1, 5.9}, "test");
    data.emplace_back("carre", 3, std::vector<double>{4.9, 5.1, 5.9, 6.1}, "test");
    data.emplace_back("carre", 4, std::vector<double>{5.2, 4.8, 6.2, 5.8}, "test");
    
    // Cluster 3 - Triangles
    data.emplace_back("triangle", 1, std::vector<double>{9.0, 9.0, 10.0, 10.0}, "test");
    data.emplace_back("triangle", 2, std::vector<double>{9.1, 8.9, 10.1, 9.9}, "test");
    data.emplace_back("triangle", 3, std::vector<double>{8.9, 9.1, 9.9, 10.1}, "test");
    data.emplace_back("triangle", 4, std::vector<double>{9.2, 8.8, 10.2, 9.8}, "test");
    
    return data;
}

// Fonction principale
int main() {
    std::cout << "=== TEST DE L'ALGORITHME K-MEANS CLUSTERING ===" << std::endl;
    
    // Création des données d'exemple
    std::vector<Image> sampleData = createSampleData();
    std::cout << "Données créées : " << sampleData.size() << " échantillons" << std::endl;
    
    // Test avec différentes valeurs de k
    for (int k = 2; k <= 4; ++k) {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "TEST AVEC K = " << k << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        KMeans kmeans(k);
        
        // Entraînement
        bool success = kmeans.fit(sampleData);
        
        if (success) {
            // Affichage des informations sur les clusters
            kmeans.displayClusterInfo(sampleData);
            
            // Calcul et affichage du score de silhouette
            double silhouetteScore = kmeans.calculateSilhouetteScore(sampleData);
            std::cout << "Score de silhouette : " << std::fixed << std::setprecision(4) << silhouetteScore << std::endl;
            
            // Interprétation du score
            if (silhouetteScore > 0.7) {
                std::cout << "Qualité de clustering : Excellente" << std::endl;
            } else if (silhouetteScore > 0.5) {
                std::cout << "Qualité de clustering : Bonne" << std::endl;
            } else if (silhouetteScore > 0.2) {
                std::cout << "Qualité de clustering : Moyenne" << std::endl;
            } else {
                std::cout << "Qualité de clustering : Faible" << std::endl;
            }
        }
    }
    
    return 0;
}
