//AIT FERHAT Thanina
//BENKERROU Lynda

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <random>
#include <cmath>
#include <stdexcept>

namespace fs = std::filesystem;

// Classe pour représenter une image avec ses métadonnées
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

// Fonction pour calculer la distance euclidienne entre deux vecteurs
double distance(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Les vecteurs doivent avoir la même taille");
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
    if (k <= 0 || k > static_cast<int>(trainingSet.size())) {
        throw std::invalid_argument("k doit être entre 1 et la taille de l'ensemble d'entraînement");
    }
    
    // Vecteur pour stocker les distances entre l'image de requête et les images d'entraînement
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

// Fonction pour calculer la matrice de confusion
std::map<std::pair<std::string, std::string>, int> calculateConfusionMatrix(
    const std::vector<Image>& testSet,
    const std::vector<Image>& trainingSet,
    int k) {
    
    std::map<std::pair<std::string, std::string>, int> confusionMatrix;

    for (const Image& testImage : testSet) {
        std::string trueClass = testImage.className;
        std::string predictedClass = predictKNN(trainingSet, testImage.values, k);
        confusionMatrix[{trueClass, predictedClass}]++;
    }

    return confusionMatrix;
}

// Calcul du taux de reconnaissance (accuracy) à partir de la matrice de confusion
double calculateAccuracy(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix) {
    int correctPredictions = 0;
    int totalPredictions = 0;

    for (const auto& entry : confusionMatrix) {
        if (entry.first.first == entry.first.second) {
            correctPredictions += entry.second;
        }
        totalPredictions += entry.second;
    }

    return totalPredictions > 0 ? static_cast<double>(correctPredictions) / totalPredictions : 0.0;
}

// Calcul du taux de confusion (confusion rate) à partir de la matrice de confusion
double calculateConfusionRate(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix) {
    return 1.0 - calculateAccuracy(confusionMatrix);
}

// Calcul le rappel pour chaque classe
std::map<std::string, double> calculateRecall(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix) {
    std::map<std::string, double> recall;
    std::map<std::string, int> truePositives;
    std::map<std::string, int> actualClassCounts;

    // Calculer les vrais positifs et le nombre total d'échantillons par classe
    for (const auto& entry : confusionMatrix) {
        actualClassCounts[entry.first.first] += entry.second;
        if (entry.first.first == entry.first.second) {
            truePositives[entry.first.first] += entry.second;
        }
    }

    // Calculer le rappel pour chaque classe
    for (const auto& classCount : actualClassCounts) {
        const std::string& className = classCount.first;
        int totalActual = classCount.second;
        int tp = truePositives[className];
        recall[className] = totalActual > 0 ? static_cast<double>(tp) / totalActual : 0.0;
    }

    return recall;
}

// Calcul la précision pour chaque classe
std::map<std::string, double> calculatePrecision(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix) {
    std::map<std::string, double> precision;
    std::map<std::string, int> truePositives;
    std::map<std::string, int> predictedClassCounts;

    // Calculer les vrais positifs et le nombre total de prédictions par classe
    for (const auto& entry : confusionMatrix) {
        predictedClassCounts[entry.first.second] += entry.second;
        if (entry.first.first == entry.first.second) {
            truePositives[entry.first.second] += entry.second;
        }
    }

    // Calculer la précision pour chaque classe
    for (const auto& classCount : predictedClassCounts) {
        const std::string& className = classCount.first;
        int totalPredicted = classCount.second;
        int tp = truePositives[className];
        precision[className] = totalPredicted > 0 ? static_cast<double>(tp) / totalPredicted : 0.0;
    }

    return precision;
}

// Calcul F-mesure
std::pair<std::map<std::string, double>, double> calculateFMeasure(
    const std::map<std::string, double>& precision, 
    const std::map<std::string, double>& recall) {
    
    std::map<std::string, double> fMeasure;
    double sumFMeasure = 0.0;
    int classCount = 0;

    for (const auto& p : precision) {
        std::string className = p.first;
        double prec = p.second;
        double rec = recall.count(className) ? recall.at(className) : 0.0;

        if (prec + rec > 0) {
            fMeasure[className] = 2 * prec * rec / (prec + rec);
            sumFMeasure += fMeasure[className];
            classCount++;
        } else {
            fMeasure[className] = 0.0;
        }
    }

    double averageFMeasure = classCount > 0 ? sumFMeasure / classCount : 0.0;
    return {fMeasure, averageFMeasure};
}

// Fonction pour diviser les données en ensembles d'entraînement et de test
std::pair<std::vector<Image>, std::vector<Image>> splitTrainTest(std::vector<Image>& allImages, double trainRatio = 0.67) {
    if (allImages.empty()) {
        return {std::vector<Image>(), std::vector<Image>()};
    }
    
    // Mélanger les données pour une division aléatoire
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(allImages.begin(), allImages.end(), g);
    
    size_t trainSize = static_cast<size_t>(allImages.size() * trainRatio);
    
    std::vector<Image> trainSet(allImages.begin(), allImages.begin() + trainSize);
    std::vector<Image> testSet(allImages.begin() + trainSize, allImages.end());
    
    return {trainSet, testSet};
}

// Fonction pour extraire le nom de classe de manière plus robuste
std::string extractClassName(const std::string& filename) {
    if (filename.length() >= 3) {
        return filename.substr(1, 2);
    }
    return "unknown";
}

// Fonction pour extraire le numéro d'échantillon de manière plus robuste
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

// Lecture des données et création des ensembles d'entraînement et de test
std::map<std::string, std::pair<std::vector<Image>, std::vector<Image>>> creationTableaux(const std::string& repertoire) {
    std::map<std::string, std::vector<Image>> allImagesByMethod;
    std::map<std::string, std::pair<std::vector<Image>, std::vector<Image>>> tableaux_fichiers;

    try {
        // Première passe : charger toutes les images
        for (const auto& entry : fs::directory_iterator(repertoire)) {
            if (entry.is_regular_file()) {
                std::vector<double> vect = readVectorsFromFolders(entry.path().string());
                
                if (vect.empty()) {
                    std::cerr << "Vecteur vide pour le fichier : " << entry.path().string() << std::endl;
                    continue;
                }

                std::string filename = entry.path().filename().string();
                std::string className = extractClassName(filename);
                int sampleNumber = extractSampleNumber(filename);
                std::string methodName = entry.path().parent_path().filename().string();

                Image img(className, sampleNumber, vect, methodName);
                allImagesByMethod[methodName].push_back(img);
            }
        }

        // Deuxième passe : diviser chaque méthode en train/test
        for (auto& methodPair : allImagesByMethod) {
            auto splitResult = splitTrainTest(methodPair.second);
            tableaux_fichiers[methodPair.first] = splitResult;
        }

    } catch (const std::exception& e) {
        std::cerr << "Erreur lors de la création des tableaux : " << e.what() << std::endl;
    }

    return tableaux_fichiers;
}

// Fonction pour afficher les résultats de manière organisée
void afficherResultats(const std::string& methodName, 
                      const std::vector<Image>& trainSet,
                      const std::vector<Image>& testSet,
                      int k) {
    
    std::cout << "\n=== Méthode : " << methodName << " (k=" << k << ") ===" << std::endl;
    std::cout << "Taille ensemble d'entraînement : " << trainSet.size() << std::endl;
    std::cout << "Taille ensemble de test : " << testSet.size() << std::endl;

    try {
        // Calcul de la matrice de confusion
        auto confusionMatrix = calculateConfusionMatrix(testSet, trainSet, k);

        // Affichage de la matrice de confusion
        std::cout << "\nMatrice de confusion :" << std::endl;
        std::cout << "Vraie_Classe\tClasse_Predite\tNombre" << std::endl;
        for (const auto& entry : confusionMatrix) {
            std::cout << entry.first.first << "\t\t" << entry.first.second << "\t\t" << entry.second << std::endl;
        }

        // Calcul et affichage des métriques
        double accuracy = calculateAccuracy(confusionMatrix);
        double confusionRate = calculateConfusionRate(confusionMatrix);
        auto recall = calculateRecall(confusionMatrix);
        auto precision = calculatePrecision(confusionMatrix);
        auto fMeasureResult = calculateFMeasure(precision, recall);

        std::cout << "\nMétriques globales :" << std::endl;
        std::cout << "Taux de reconnaissance (Accuracy) : " << accuracy * 100.0 << "%" << std::endl;
        std::cout << "Taux de confusion : " << confusionRate * 100.0 << "%" << std::endl;
        std::cout << "F-mesure moyenne : " << fMeasureResult.second * 100.0 << "%" << std::endl;

        std::cout << "\nMétriques par classe :" << std::endl;
        std::cout << "Classe\tRappel\tPrécision\tF-mesure" << std::endl;
        for (const auto& classRecall : recall) {
            const std::string& className = classRecall.first;
            double rec = classRecall.second * 100.0;
            double prec = precision.count(className) ? precision.at(className) * 100.0 : 0.0;
            double fm = fMeasureResult.first.count(className) ? fMeasureResult.first.at(className) * 100.0 : 0.0;
            
            std::cout << className << "\t" << rec << "%\t" << prec << "%\t\t" << fm << "%" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Erreur lors du calcul pour k=" << k << " : " << e.what() << std::endl;
    }
}

int main() {
    // Chemins des dossiers (à adapter selon votre environnement)
    std::vector<std::string> chemins_dossiers = {
        ""
    };

    // Traitement de chaque dossier/méthode
    for (const std::string& repertoire : chemins_dossiers) {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Traitement du répertoire : " << repertoire << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        auto tableaux_fichiers = creationTableaux(repertoire);

        if (tableaux_fichiers.empty()) {
            std::cerr << "Aucune donnée trouvée dans : " << repertoire << std::endl;
            continue;
        }

        // Pour chaque méthode trouvée
        for (const auto& method_data : tableaux_fichiers) {
            const std::string& methodName = method_data.first;
            const auto& trainSet = method_data.second.first;
            const auto& testSet = method_data.second.second;

            if (trainSet.empty() || testSet.empty()) {
                std::cerr << "Ensemble d'entraînement ou de test vide pour la méthode : " << methodName << std::endl;
                continue;
            }

            // Test avec différentes valeurs de k
            std::cout << "\n--- Résultats pour la méthode : " << methodName << " ---" << std::endl;
            
            for (int k = 1; k <= std::min(10, static_cast<int>(trainSet.size())); ++k) {
                afficherResultats(methodName, trainSet, testSet, k);
            }
        }
    }

    std::cout << "\nTraitement terminé." << std::endl;
    return 0;
}
