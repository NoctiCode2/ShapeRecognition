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
#include <stdexcept>

namespace fs = std::filesystem;

// Classe représentant une image avec ses caractéristiques.
class Image {
public:
    std::string className;      // Nom de la classe de l'image.
    int sampleNumber;          // Numéro de l'échantillon.
    std::vector<double> values; // Vecteur de caractéristiques de l'image.
    std::string methodName;    // Nom de la méthode utilisée pour traiter l'image.
    
    // Constructeur de la classe Image.
    Image(std::string className, int sampleNumber, const std::vector<double>& values, std::string methodName)
            : className(std::move(className)), sampleNumber(sampleNumber), values(values), methodName(std::move(methodName)) {}
};

// Classe implémentant l'algorithme KMeans.
class KMeans {
public:
    KMeans(int k, int maxIterations = 100) : k(k), maxIterations(maxIterations) {
        if (k <= 0) {
            throw std::invalid_argument("Le nombre de clusters k doit être positif");
        }
    }
    
    const std::vector<int>& getAssignments() const { return assignments; }
    const std::vector<std::vector<double>>& getCentroids() const { return centroids; }
    int getK() const { return k; }
    int getIterations() const { return iterations; }

    // Assigner une image à un cluster en trouvant le centroïde le plus proche.
    int assignCluster(const Image& image) {
        if (centroids.empty()) {
            throw std::runtime_error("Le modèle n'a pas été entraîné");
        }
        return findClosestCentroid(image.values);
    }

    // Calculer le score de silhouette pour évaluer la qualité du clustering.
    double calculateSilhouetteScore(const std::vector<Image>& images) {
        if (images.empty() || assignments.empty()) {
            return 0.0;
        }

        std::vector<double> silhouetteScores(images.size(), 0.0);

        for (size_t i = 0; i < images.size(); ++i) {
            double a = calculateIntraClusterDistance(images, i);
            double b = calculateNearestClusterDistance(images, i);
            
            if (a == 0.0 && b == 0.0) {
                silhouetteScores[i] = 0.0;
            } else {
                silhouetteScores[i] = (b - a) / std::max(a, b);
            }
        }

        // Calculer le score moyen
        double sum = 0.0;
        int validCount = 0;
        for (double score : silhouetteScores) {
            if (!std::isinf(score) && !std::isnan(score)) {
                sum += score;
                validCount++;
            }
        }
        return validCount > 0 ? sum / validCount : 0.0;
    }

    // Calculer l'inertie (Within-Cluster Sum of Squares) pour la méthode Elbow
    double calculateInertia(const std::vector<Image>& images) {
        if (images.empty() || assignments.empty() || centroids.empty()) {
            return 0.0;
        }

        double inertia = 0.0;
        for (size_t i = 0; i < images.size(); ++i) {
            int clusterIdx = assignments[i];
            if (clusterIdx >= 0 && clusterIdx < static_cast<int>(centroids.size())) {
                double dist = euclideanDistance(images[i].values, centroids[clusterIdx]);
                inertia += dist * dist; // Somme des carrés des distances
            }
        }
        return inertia;
    }

    // Exécuter l'algorithme KMeans sur les images.
    bool fit(const std::vector<Image>& images) {
        if (images.empty()) {
            throw std::invalid_argument("Le vecteur d'images ne peut pas être vide");
        }
        
        if (k > static_cast<int>(images.size())) {
            throw std::invalid_argument("Le nombre de clusters ne peut pas être supérieur au nombre d'images");
        }

        // Vérifier que toutes les images ont la même dimension
        size_t dimension = images[0].values.size();
        for (const auto& img : images) {
            if (img.values.size() != dimension) {
                throw std::invalid_argument("Toutes les images doivent avoir la même dimension");
            }
        }

        // Initialiser les centres de clusters
        initCentroids(images);
        assignments.assign(images.size(), -1);
        
        bool converged = false;
        iterations = 0;

        while (!converged && iterations < maxIterations) {
            std::vector<int> newAssignments(images.size());
            bool changed = assignClusters(images, newAssignments);
            
            if (!changed) {
                converged = true;
            } else {
                assignments = newAssignments;
                recalculateCentroids(images, assignments);
                iterations++;
            }
        }

        return converged;
    }

private:
    int k;                                          // Nombre de clusters.
    int maxIterations;                              // Nombre maximum d'itérations.
    int iterations = 0;                             // Nombre d'itérations effectuées.
    std::vector<std::vector<double>> centroids;     // Centroïdes des clusters.
    std::vector<int> assignments;                   // Assignations finales des clusters.

    // Calculer la distance intra-cluster moyenne pour un point
    double calculateIntraClusterDistance(const std::vector<Image>& images, size_t pointIndex) {
        int clusterIdx = assignments[pointIndex];
        double sum = 0.0;
        int count = 0;

        for (size_t i = 0; i < images.size(); ++i) {
            if (i != pointIndex && assignments[i] == clusterIdx) {
                sum += euclideanDistance(images[pointIndex].values, images[i].values);
                count++;
            }
        }

        return count > 0 ? sum / count : 0.0;
    }

    // Calculer la distance moyenne au cluster le plus proche
    double calculateNearestClusterDistance(const std::vector<Image>& images, size_t pointIndex) {
        int currentCluster = assignments[pointIndex];
        double minDistance = std::numeric_limits<double>::max();

        for (int clusterIdx = 0; clusterIdx < k; ++clusterIdx) {
            if (clusterIdx == currentCluster) continue;

            double sum = 0.0;
            int count = 0;

            for (size_t i = 0; i < images.size(); ++i) {
                if (assignments[i] == clusterIdx) {
                    sum += euclideanDistance(images[pointIndex].values, images[i].values);
                    count++;
                }
            }

            if (count > 0) {
                double avgDistance = sum / count;
                minDistance = std::min(minDistance, avgDistance);
            }
        }

        return minDistance == std::numeric_limits<double>::max() ? 0.0 : minDistance;
    }

    // Initialiser les centres des clusters avec K-means++
    void initCentroids(const std::vector<Image>& images) {
        centroids.clear();
        centroids.reserve(k);

        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Choisir le premier centroïde aléatoirement
        std::uniform_int_distribution<> dis(0, images.size() - 1);
        centroids.push_back(images[dis(gen)].values);

        // Choisir les centroïdes suivants avec K-means++
        for (int i = 1; i < k; ++i) {
            std::vector<double> distances(images.size());
            double totalDistance = 0.0;

            // Calculer la distance au centroïde le plus proche pour chaque point
            for (size_t j = 0; j < images.size(); ++j) {
                double minDist = std::numeric_limits<double>::max();
                for (const auto& centroid : centroids) {
                    double dist = euclideanDistance(images[j].values, centroid);
                    minDist = std::min(minDist, dist);
                }
                distances[j] = minDist * minDist; // Carré de la distance
                totalDistance += distances[j];
            }

            // Choisir le prochain centroïde avec une probabilité proportionnelle à la distance
            std::uniform_real_distribution<> realDis(0.0, totalDistance);
            double target = realDis(gen);
            double cumSum = 0.0;

            for (size_t j = 0; j < images.size(); ++j) {
                cumSum += distances[j];
                if (cumSum >= target) {
                    centroids.push_back(images[j].values);
                    break;
                }
            }
        }
    }

    // Assigner chaque image à un cluster.
    bool assignClusters(const std::vector<Image>& images, std::vector<int>& newAssignments) {
        bool changed = false;
        for (size_t i = 0; i < images.size(); ++i) {
            int closest = findClosestCentroid(images[i].values);
            newAssignments[i] = closest;
            if (assignments.empty() || closest != assignments[i]) {
                changed = true;
            }
        }
        return changed;
    }

    // Recalculer les centres des clusters après l'assignation des images.
    void recalculateCentroids(const std::vector<Image>& images, const std::vector<int>& assignments) {
        if (images.empty()) return;

        size_t dimension = images[0].values.size();
        std::vector<std::vector<double>> sums(k, std::vector<double>(dimension, 0.0));
        std::vector<int> counts(k, 0);

        for (size_t i = 0; i < images.size(); ++i) {
            int clusterIdx = assignments[i];
            if (clusterIdx >= 0 && clusterIdx < k) {
                for (size_t j = 0; j < dimension; ++j) {
                    sums[clusterIdx][j] += images[i].values[j];
                }
                counts[clusterIdx]++;
            }
        }

        for (int i = 0; i < k; ++i) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < dimension; ++j) {
                    centroids[i][j] = sums[i][j] / counts[i];
                }
            }
            // Si un cluster est vide, on garde l'ancien centroïde
        }
    }

    // Trouver le centroïde le plus proche d'une image.
    int findClosestCentroid(const std::vector<double>& values) {
        double minDistance = std::numeric_limits<double>::max();
        int closest = 0;

        for (int i = 0; i < k; ++i) {
            double distance = euclideanDistance(values, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                closest = i;
            }
        }

        return closest;
    }

    // Calculer la distance euclidienne entre deux vecteurs.
    static double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Les vecteurs doivent avoir la même taille");
        }

        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

// Lire des vecteurs de données depuis des fichiers et les stocker dans un vecteur.
std::vector<double> readVectorsFromFolders(const std::string& folderName) {
    std::ifstream file(folderName);
    std::vector<double> vect;

    if (file.is_open()) {
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

// Fonction pour extraire le nom de classe de manière robuste
std::string extractClassName(const std::string& filename) {
    if (filename.length() >= 3) {
        return filename.substr(1, 2);
    }
    return "unknown";
}

// Fonction pour extraire le numéro d'échantillon de manière robuste
int extractSampleNumber(const std::string& filename) {
    try {
        if (filename.length() >= 7) {
            return std::stoi(filename.substr(4, 3));
        }
    } catch (const std::exception&) {
        std::cerr << "Erreur lors de l'extraction du numéro d'échantillon de : " << filename << std::endl;
    }
    return 0;
}

// Charger des images depuis un dossier et les stocker dans un vecteur d'images.
std::vector<Image> chargeImages(const std::string& repertoire) {
    std::vector<Image> images;
    
    try {
        for (const auto& entry : fs::directory_iterator(repertoire)) {
            if (entry.is_regular_file()) {
                std::string fichier = entry.path().filename().string();
                std::vector<double> vector = readVectorsFromFolders(entry.path().string());
                
                if (vector.empty()) {
                    std::cerr << "Vecteur vide pour le fichier : " << fichier << std::endl;
                    continue;
                }

                std::string className = extractClassName(fichier);
                int sampleNumber = extractSampleNumber(fichier);
                std::string methodName = entry.path().parent_path().filename().string();

                images.emplace_back(className, sampleNumber, vector, methodName);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors du chargement des images : " << e.what() << std::endl;
    }

    return images;
}

// Affecter des classes aux clusters et analyser la répartition
void analyzeClusterComposition(const std::vector<Image>& images, const std::vector<int>& clusterAssignments, int k) {
    std::vector<std::unordered_map<std::string, int>> classCountsInClusters(k);
    std::vector<int> clusterSizes(k, 0);

    // Compter les occurrences de chaque classe dans chaque cluster
    for (size_t i = 0; i < images.size(); ++i) {
        int clusterIndex = clusterAssignments[i];
        if (clusterIndex >= 0 && clusterIndex < k) {
            classCountsInClusters[clusterIndex][images[i].className]++;
            clusterSizes[clusterIndex]++;
        }
    }

    // Afficher les résultats détaillés
    std::cout << "\n=== Composition des clusters ===" << std::endl;
    for (int i = 0; i < k; ++i) {
        std::cout << "Cluster " << i << " (Taille: " << clusterSizes[i] << "):" << std::endl;
        
        if (clusterSizes[i] == 0) {
            std::cout << "  Cluster vide" << std::endl;
            continue;
        }

        // Trouver la classe dominante
        std::string dominantClass;
        int maxCount = 0;
        for (const auto& pair : classCountsInClusters[i]) {
            std::cout << "  Classe " << pair.first << ": " << pair.second 
                      << " occurrences (" << (100.0 * pair.second / clusterSizes[i]) << "%)" << std::endl;
            if (pair.second > maxCount) {
                maxCount = pair.second;
                dominantClass = pair.first;
            }
        }
        
        double purity = 100.0 * maxCount / clusterSizes[i];
        std::cout << "  Classe dominante: " << dominantClass << " (Pureté: " << purity << "%)" << std::endl;
        std::cout << std::endl;
    }
}

// Calculer la pureté globale du clustering
double calculateGlobalPurity(const std::vector<Image>& images, const std::vector<int>& clusterAssignments, int k) {
    std::vector<std::unordered_map<std::string, int>> classCountsInClusters(k);
    
    for (size_t i = 0; i < images.size(); ++i) {
        int clusterIndex = clusterAssignments[i];
        if (clusterIndex >= 0 && clusterIndex < k) {
            classCountsInClusters[clusterIndex][images[i].className]++;
        }
    }

    int totalCorrect = 0;
    for (int i = 0; i < k; ++i) {
        int maxCount = 0;
        for (const auto& pair : classCountsInClusters[i]) {
            maxCount = std::max(maxCount, pair.second);
        }
        totalCorrect += maxCount;
    }

    return images.empty() ? 0.0 : (100.0 * totalCorrect / images.size());
}

// Programme principal
int main() {
    std::vector<std::string> chemins_dossiers = {
        
    };

    for (const std::string& repertoire : chemins_dossiers) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Traitement du répertoire : " << repertoire << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        auto images = chargeImages(repertoire);
        
        if (images.empty()) {
            std::cerr << "Aucune image trouvée dans : " << repertoire << std::endl;
            continue;
        }

        std::cout << "Nombre d'images chargées : " << images.size() << std::endl;
        
        // Compter les classes uniques
        std::unordered_map<std::string, int> classCount;
        for (const auto& img : images) {
            classCount[img.className]++;
        }
        std::cout << "Nombre de classes uniques : " << classCount.size() << std::endl;
        
        std::cout << "\nRépartition des classes :" << std::endl;
        for (const auto& pair : classCount) {
            std::cout << "  Classe " << pair.first << ": " << pair.second << " images" << std::endl;
        }

        // Test avec différentes valeurs de k
        std::vector<double> inerties;
        std::vector<double> silhouetteScores;

        std::cout << "\n--- Analyse K-means ---" << std::endl;
        std::cout << "k\tInertie\t\tSilhouette\tPureté(%)\tItérations\tConvergé" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        for (int k = 1; k <= std::min(10, static_cast<int>(images.size())); ++k) {
            try {
                KMeans km(k, 300); // Augmenter le nombre max d'itérations
                bool converged = km.fit(images);

                double inertia = km.calculateInertia(images);
                double silhouetteScore = (k > 1) ? km.calculateSilhouetteScore(images) : 0.0;
                double purity = calculateGlobalPurity(images, km.getAssignments(), k);

                inerties.push_back(inertia);
                silhouetteScores.push_back(silhouetteScore);

                std::cout << k << "\t" << std::fixed << std::setprecision(2) 
                          << inertia << "\t\t" << silhouetteScore << "\t\t" 
                          << purity << "\t\t" << km.getIterations() << "\t\t" 
                          << (converged ? "Oui" : "Non") << std::endl;

                // Affichage détaillé pour quelques valeurs de k intéressantes
                if (k == 2 || k == 3 || k == static_cast<int>(classCount.size())) {
                    std::cout << "\n--- Détails pour k=" << k << " ---" << std::endl;
                    analyzeClusterComposition(images, km.getAssignments(), k);
                }

            } catch (const std::exception& e) {
                std::cerr << "Erreur pour k=" << k << " : " << e.what() << std::endl;
            }
        }

        // Suggestions basées sur les métriques
        std::cout << "\n=== Recommandations ===" << std::endl;
        
        // Méthode du coude pour l'inertie
        if (inerties.size() >= 3) {
            std::cout << "Méthode du coude (Inertie) : Analyser le graphique pour détecter le 'coude'" << std::endl;
        }

        // Meilleur score de silhouette
        if (!silhouetteScores.empty()) {
            auto maxSilIt = std::max_element(silhouetteScores.begin(), silhouetteScores.end());
            if (maxSilIt != silhouetteScores.end()) {
                int bestK = std::distance(silhouetteScores.begin(), maxSilIt) + 2; // +2 car on commence à k=2 pour silhouette
                std::cout << "Meilleur score de silhouette : k=" << bestK 
                          << " (score=" << *maxSilIt << ")" << std::endl;
            }
        }

        std::cout << "Nombre de classes réelles : " << classCount.size() 
                  << " (à comparer avec k optimal)" << std::endl;
    }

    std::cout << "\nTraitement terminé." << std::endl;
    return 0;
}
