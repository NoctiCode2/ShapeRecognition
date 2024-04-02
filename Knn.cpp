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
//  fonction pour calculer la distance euclidienne entre deux vecteurs
double distance(const std::vector<double>& v1, const std::vector<double>& v2) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

//  fonction pour prédire la classe d'une image en utilisant k-NN
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

//fonction pour calculer la matrice de confusion
std::map<std::pair<std::string, std::string>, int> calculateConfusionMatrix(const std::vector<Image>& testSet,const std::vector<Image>& trainingSet,int k) {
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

    return static_cast<double>(correctPredictions) / totalPredictions;
}

// Calcul du taux de confusion (confusion rate) à partir de la matrice de confusion
double calculateConfusionRate(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix) {
    int totalIncorrectPredictions = 0;
    int totalPredictions = 0;

    for (const auto& entry : confusionMatrix) {
        if (entry.first.first != entry.first.second) {
            totalIncorrectPredictions += entry.second;
        }
        totalPredictions += entry.second;
    }

    return static_cast<double>(totalIncorrectPredictions) / totalPredictions;
}
// Calcul le rappel pour chaque classe
std::map<std::string, double> calculateRecall(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix) {
    std::map<std::string, double> recall;
    std::map<std::string, int> truePositives;
    std::map<std::string, int> falseNegatives;

    for (const auto& entry : confusionMatrix) {
        if (entry.first.first == entry.first.second) {
            truePositives[entry.first.first] += entry.second;
        } else {
            falseNegatives[entry.first.first] += entry.second;
        }
    }

    for (const auto& tp : truePositives) {
        double denom = tp.second + falseNegatives[tp.first];
        recall[tp.first] = denom != 0 ? static_cast<double>(tp.second) / denom : 0;
    }

    return recall;
}

// Calcul la précision pour chaque classe
std::map<std::string, double> calculatePrecision(const std::map<std::pair<std::string, std::string>, int>& confusionMatrix) {
    std::map<std::string, double> precision;
    std::map<std::string, int> truePositives;
    std::map<std::string, int> falsePositives;

    for (const auto& entry : confusionMatrix) {
        if (entry.first.first == entry.first.second) {
            truePositives[entry.first.second] += entry.second;
        } else {
            falsePositives[entry.first.second] += entry.second;
        }
    }

    for (const auto& tp : truePositives) {
        double denom = tp.second + falsePositives[tp.first];
        precision[tp.first] = denom != 0 ? static_cast<double>(tp.second) / denom : 0;
    }

    return precision;
}
//calcul F-mesure
std::pair<std::map<std::string, double>, double> calculateFMeasure(const std::map<std::string, double>& precision, const std::map<std::string, double>& recall) {
    std::map<std::string, double> fMeasure;
    double sumFMeasure = 0.0;
    int classCount = 0;

    for (const auto& p : precision) {
        std::string className = p.first;
        double prec = p.second;
        double rec = recall.at(className);

        if (prec + rec != 0) {
            fMeasure[className] = 2 * prec * rec / (prec + rec);
            sumFMeasure += fMeasure[className];
            classCount++;
        } else {
            fMeasure[className] = 0;
        }
    }

    double averageFMeasure = classCount > 0 ? sumFMeasure / classCount : 0;
    return {fMeasure, averageFMeasure};
}





//Lecture des données
std::map<std::string, std::pair<std::vector<Image>, std::vector<Image>>> creationTableaux(const std::string& repertoire) {
    std::map<std::string, std::pair<std::vector<Image>, std::vector<Image>>> tableaux_fichiers;

    try {
        for (const auto& entry : fs::directory_iterator(repertoire)) {
            std::string fichier = entry.path().filename().string();
            if (entry.is_regular_file()) {
                std::vector<double> vect = readVectorsFromFolders(entry.path().string());

                std::string className = entry.path().filename().string().substr(1, 2); // Extraire le nom de la classe
                int sampleNumber = std::stoi(entry.path().filename().string().substr(4, 3)); // Extraire le numéro de l'échantillon
                std::string methodName = entry.path().extension().string(); // Obtenir l'extension du fichier (E34)

                Image img(className, sampleNumber, vect, methodName);

                // Stocker l'image dans la map
                if (tableaux_fichiers.find(methodName) == tableaux_fichiers.end()) {
                    tableaux_fichiers[methodName] = std::make_pair(std::vector<Image>(), std::vector<Image>());
                }

                // Séparation des données pour l'ensemble test et entraînement
                if (tableaux_fichiers[methodName].first.size() < (tableaux_fichiers[methodName].second.size() * 2 / 3)) {
                    tableaux_fichiers[methodName].first.push_back(img); // Ajouter à l'ensemble de test
                } else {
                    tableaux_fichiers[methodName].second.push_back(img); // Ajouter à l'ensemble d'entraînement
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors de la création des tableaux : " << e.what() << std::endl;
    }

    return tableaux_fichiers;
}





int main() {

    std::vector<std::string> chemins_dossiers = {
            "C:/Users/AIT FERHAT/Desktop/Mes cours/RF/ProjectRF/Nouveau dossier/E34",
            "C:/Users/AIT FERHAT/Desktop/Mes cours/RF/ProjectRF/Nouveau dossier/F0",
            "C:/Users/AIT FERHAT/Desktop/Mes cours/RF/ProjectRF/Nouveau dossier/GFD",
            "C:/Users/AIT FERHAT/Desktop/Mes cours/RF/ProjectRF/Nouveau dossier/SA"
    };

    for (const std::string &repertoire: chemins_dossiers) {
        auto tableaux_fichiers = creationTableaux(repertoire);


        //affichage des détails de chaque image pour les ensembles d'entraînement et de test
        for (const auto &method_data: tableaux_fichiers) {
            std::cout << "Méthode : " << method_data.first << std::endl;

            // Affichage des détails pour l'ensemble de test
            std::cout << "Donnees de test :" << std::endl;
            for (const Image &img: method_data.second.first) {
                std::cout << "  Num echantillon : " << img.sampleNumber << std::endl;
                std::cout << "Methode : " << img.methodName << std::endl;
                std::cout << "  Vecteur de valeurs : ";
                for (double value: img.values) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            }

            // Affichage des détails pour l'ensemble d'entraînement
            std::cout << "Donnees entrainement :" << std::endl;
            for (const Image &img: method_data.second.second) {
                std::cout << "  Num echantillon : " << img.sampleNumber << std::endl;
                std::cout << "Methode : " << img.methodName << std::endl;

                std::cout << "  Vecteur de valeurs : ";
                for (double value: img.values) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            }

            // Utilisation de la fonction predictKNN pour prédire la classe des images de test
            for (int k = 1; k <= 10; ++k) { // le nombre de voisins k
                for (const auto &method_data: tableaux_fichiers) {


                    //  prédictions pour l'ensemble de test
                    std::cout << "Predictions pour ensemble de test :" << std::endl;
                    for (const Image &img: method_data.second.first) {
                        std::string predictedClass = predictKNN(method_data.second.second, img.values, k);
                        std::cout << "Valeur K :" << k << std::endl;
                        std::cout << "Num echantillon : " << img.sampleNumber << std::endl;
                        std::cout << "Methode : " << img.methodName << std::endl;
                        std::cout << "  Vecteur de valeurs : ";
                        for (double value: img.values) {
                            std::cout << value << " ";
                        }
                        std::cout << std::endl;
                        std::cout << "Classe predite : " << predictedClass << std::endl;
                        std::cout << std::endl;


                    }

                    std::map<std::pair<std::string, std::string>, int> confusionMatrix = calculateConfusionMatrix(
                            method_data.second.first, // Ensemble de test
                            method_data.second.second, // Ensemble d'entraînement
                            k
                    );

                    // Affichage de la matrice de confusion
                    std::cout << "Matrice de confusion :" << std::endl;
                    //NbOcc est le nombre d'occurrences où la vraie classe a été prédite comme étant la classe prédite.
                    //V_C pour Vraie Classe et P_C pour Classe prédite
                    std::cout << "V_C \t P_C \t NbOcc " << std::endl;
                    for (const auto &entry: confusionMatrix) {
                        std::cout << entry.first.first << "\t" << entry.first.second << "\t" << entry.second
                                  << std::endl;
                    }

                    // Taux de reconnaissance (accuracy)
                    double accuracy = calculateAccuracy(confusionMatrix);

                    // Taux de confusion (confusion rate)
                    double confusionRate = calculateConfusionRate(confusionMatrix);

                    // Affichage des statistiques
                    std::cout << "Taux de reconnaissance (Accuracy) : " << accuracy * 100.0 << "%" << std::endl;
                    std::cout << "Taux de confusion (Confusion Rate) : " << confusionRate * 100.0 << "%" << std::endl;

                    // Calcul le rappel et la précision
                    std::map<std::string, double> recall = calculateRecall(confusionMatrix);
                    std::map<std::string, double> precision = calculatePrecision(confusionMatrix);

                    // Affichage du rappel et de la précision pour chaque classe
                    std::cout << "Rappel et Precision par classe :" << std::endl;
                    std::cout << "Valeur K :" << k << std::endl;
                    for (const auto &classRecall: recall) {

                        std::cout << "Classe " << classRecall.first << std::endl;
                        std::cout << "Rappel :" << classRecall.second * 100.0 << "%" << std::endl;
                        std::cout << "Precision: " << precision[classRecall.first] * 100.0 << "%" << std::endl;
                    }
                    auto fMeasureResult = calculateFMeasure(precision, recall);
                    auto fMeasure = fMeasureResult.first;
                    double averageFMeasure = fMeasureResult.second;

                    // Affichage de la F-mesure pour chaque classe
                    std::cout << "F-mesure par classe :" << std::endl;
                    for (const auto &fm: fMeasure) {
                        std::cout << "Classe " << fm.first << " : " << fm.second * 100.0 << "%" << std::endl;
                    }

                    // Affichage de la F-mesure moyenne
                    std::cout << "F-mesure moyenne sur toutes les classes : " << averageFMeasure * 100.0 << "%"
                              << std::endl;
                }

            }

            return 0;
        }
    }
}