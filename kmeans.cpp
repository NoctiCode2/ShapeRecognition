//AIT FERHAT Thanina
//BENKERROU Lynda

#include "kmeans.h"
#include <random>
#include <utility>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <limits>

namespace fs = std::filesystem;

// Implémentation du constructeur de KMeans
KMeans::KMeans(int k) : k(k), centroids(), assignments() {}

// Retourne les affectations de cluster
const std::vector<int>& KMeans::getAssignments() const { 
    return assignments; 
}

// Retourne le nombre de clusters
int KMeans::getK() const { 
    return k; 
}

// Assigner une image à un cluster en trouvant le centroïde le plus proche
int KMeans::assignCluster(const Image& image) {
    int closest = findClosestCentroid(image.values);
    return closest;
}

// Calculer le score de silhouette pour évaluer la qualité du clustering
double KMeans::calculateSilhouetteScore(const std::vector<Image>& images) {
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
            for (int k = 0; k < this->k; ++k) {
                if (k != assignments[i]) {
                    double tempB = 0.0;
                    int countB = 0;
                    for (size_t j = 0; j < images.size(); ++j) {
                        if (assignments[j] == k) {
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

            // Si a ou b est infini ou NaN, définir le score à 0 ou un autre valeur par défaut
            if (a == std::numeric_limits<double>::infinity() || b == std::numeric_limits<double>::infinity() || std::isnan(a) || std::isnan(b)) {
                silhouetteScores[i] = 0.0;
            } else {
                silhouetteScores[i] = (b - a) / std::max(a, b);
            }
        }

        // Calculer le score moyen en excluant les valeurs infinies ou NaN
        double sum = 0.0;
        int count = 0;
        for (double score : silhouetteScores) {
            if (!std::isinf(score) && !std::isnan(score)) {
                sum += score;
                count++;
            }
        }
    return count > 0 ? sum / count : 0;
}

// Calculer l'inertie du clustering (Méthode Elbow)
double KMeans::calculateInertie(const std::vector<Image>& images) {
    double inertie = 0.0;
    for (size_t i = 0; i < images.size(); ++i) {
        int clusterIdx = assignments[i];
        inertie += euclideanDistance(images[i].values, centroids[clusterIdx]);
    }
    return inertie;
}

// Exécuter l'algorithme KMeans sur les images
void KMeans::fit(const std::vector<Image>& images) {
        // Initialiser les centres de clusters
        initCentroids(images);

        std::vector<int> assignments(images.size());
        bool changed;

        do {
            changed = assignClusters(images, assignments);
            recalculateCentroids(images, assignments);
        } while (changed);
    this->assignments = assignments; // Stocker les assignations finales
}

// Initialiser les centres des clusters de manière aléatoire
void KMeans::initCentroids(const std::vector<Image>& images) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, images.size() - 1);

        centroids.clear();
        for (int i = 0; i < k; ++i) {
            centroids.push_back(images[dis(gen)].values);
    }
}

// Assigner chaque image à un cluster
bool KMeans::assignClusters(const std::vector<Image>& images, std::vector<int>& assignments) {
        bool changed = false;
        for (size_t i = 0; i < images.size(); ++i) {
            int closest = findClosestCentroid(images[i].values);
            if (closest != assignments[i]) {
                assignments[i] = closest;
                changed = true;
            }
    }
    return changed;
}

// Recalculer les centres des clusters après l'assignation des images
void KMeans::recalculateCentroids(const std::vector<Image>& images, const std::vector<int>& assignments) {
        std::vector<std::vector<double>> sums(k, std::vector<double>(images[0].values.size(), 0.0));
        std::vector<int> counts(k, 0);

        for (size_t i = 0; i < images.size(); ++i) {
            for (size_t j = 0; j < images[i].values.size(); ++j) {
                sums[assignments[i]][j] += images[i].values[j];
            }
            counts[assignments[i]]++;
        }

        for (int i = 0; i < k; ++i) {
            if (counts[i] == 0) continue;
            for (size_t j = 0; j < sums[i].size(); ++j) {
                centroids[i][j] = sums[i][j] / counts[i];
        }
    }
}

// Trouver le centroïde le plus proche d'une image
int KMeans::findClosestCentroid(const std::vector<double>& values) {
        double min_distance = std::numeric_limits<double>::max();
        int closest = -1;

        for (int i = 0; i < k; ++i) {
            double distance = euclideanDistance(values, centroids[i]);
            if (distance < min_distance) {
                min_distance = distance;
                closest = i;
            }
    }

    return closest;
}

// Calculer la distance euclidienne entre deux vecteurs
double KMeans::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// Charger des images depuis un dossier et les stocker dans un vecteur d'images.
std::vector<Image> chargeImages(const std::string& repertoire) {
    std::vector<Image> images;
    for (const auto& entry : fs::directory_iterator(repertoire)) {
        std::string fichier = entry.path().filename().string();
        if (entry.is_regular_file()) {
            std::vector<double> vector = readVectorsFromFolders(entry.path().string());
            std::string className = fichier.substr(1, 2); // Extraire le nom de la classe
            int sampleNumber = std::stoi(fichier.substr(4, 3)); // Extraire le numéro de l'échantillon
            std::string methodName = entry.path().extension().string(); // Obtenir l'extension du fichier

            images.emplace_back(className, sampleNumber, vector, methodName);
        }
    }
    return images;
}

// Affecter des classes aux clusters
void assignClassesToClusters(const std::vector<Image>& images, const std::vector<int>& clusterAssignments, int k) {
    std::vector<std::unordered_map<std::string, int>> classCountsInClusters(k);

    for (size_t i = 0; i < images.size(); ++i) {
        int clusterIndex = clusterAssignments[i];
        classCountsInClusters[clusterIndex][images[i].className]++;
    }

    // Afficher les résultats
    for (int i = 0; i < k; ++i) {
        std::cout << "Cluster " << i << ":" << std::endl;
        for (const auto& pair : classCountsInClusters[i]) {
            std::cout << "  Classe " << pair.first << ": " << pair.second << " occurrences" << std::endl;
        }
    }
}

//Programme principal
#ifndef KMEANS_LIB
int main() {

    std::vector<std::string> chemins_dossiers = {
            //"C:/Users/AIT FERHAT/Desktop/Mes cours/RF/ProjectRF/Nouveau dossier/E34",
            //"C:/Users/AIT FERHAT/Desktop/Mes cours/RF/ProjectRF/Nouveau dossier/F0",
            "C:/Users/AIT FERHAT/Desktop/Mes cours/RF/ProjectRF/Nouveau dossier/GFD",
            //"C:/Users/AIT FERHAT/Desktop/Mes cours/RF/ProjectRF/Nouveau dossier/SA"
    };

    for (const std::string& repertoire : chemins_dossiers) {
        auto images = chargeImages(repertoire);

        //std::vector<double> inerties;
        for (int k = 1; k <= 10; ++k) {
            KMeans km(k); // Nombre de clusters
            km.fit(images); // Exécution de K-means sur les images d'entraînement

            //double inertie = km.calculateInertie(images);
            //inerties.push_back(inertie);
            //std::cout << "Inertie pour K=" << k << " : " << inertie << std::endl;

            double silhouetteScore = km.calculateSilhouetteScore(images);
            std::cout << "Score de silhouette moyen: " << silhouetteScore << std::endl;
            assignClassesToClusters(images, km.getAssignments(), km.getK());
        }


    }
    return 0;
}
#endif // KMEANS_LIB