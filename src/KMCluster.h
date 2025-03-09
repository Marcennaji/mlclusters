// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text
// of which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or
// see the "LICENSE" file for more details.

#pragma once

#ifdef _MSC_VER
#pragma warning(                                                               \
    disable : 4244) // conversions numeriques entre longint et double
#endif

#include "KMClusterInstance.h"
#include "KMParameters.h"
#include "KWAttributeStats.h"
#include "KWObject.h"
#include "Object.h"

/// Classe Cluster kmean, chargée de la gestion des instances, des centroides,
/// du calcul des distances, et des statistiques afferentes

class KMCluster : public NumericKeyDictionary {

public:
  KMCluster(const KMParameters *params);
  ~KMCluster(void);

  /** ajout d'une instance au cluster */
  void AddInstance(KWObject *);

  /** (re) initialise les stats, sans toucher aux eventuels centroides existants
   */
  void InitializeStatistics();

  /** suppression d'une instance du cluster */
  KWObject *RemoveInstance(KWObject *);

  /** obtenir l'instance la plus proche du centroide */
  const KMClusterInstance *GetInstanceNearestToCentroid() const;

  /** obtenir l'instance la plus eloignee du centroide */
  const KMClusterInstance *GetInstanceFurthestToCentroid() const;

  /** obtenir le parametrage utilisé pour le clustering */
  const KMParameters *GetParameters() const;

  /** fournir le parametrage utilisé pour le clustering */
  void SetParameters(const KMParameters *);

  /** obtenir le tableau des valeurs du centre virtuel du cluster (le centroide
   * du modele) */
  const ContinuousVector &GetModelingCentroidValues() const;

  /** obtenir le tableau des valeurs du centre virtuel du cluster obtenu lors de
   * la phase d'evaluation */
  const ContinuousVector &GetEvaluationCentroidValues() const;

  /** obtenir le tableau des valeurs initiales (avant convergence) du centre
   * virtuel du cluster */
  const ContinuousVector &GetInitialCentroidValues() const;

  /** initialiser le tableau des valeurs du centre virtuel du cluster (a partir
   * des valeurs lues dans un dictionnaire, par exemple) */
  void SetModelingCentroidValues(const ContinuousVector &);

  /** initialiser la valeur de depart d'un centroide */
  void SetInitialCentroidValues(const ContinuousVector &);

  const ALString GetClassLabel() const;

  /** obtenir la distance moyenne des instances vis a vis du centre du cluster
   */
  const Continuous GetMeanDistance(KMParameters::DistanceType) const;

  /** obtenir la distance totale des instances vis a vis du centre du cluster */
  const Continuous GetDistanceSum(KMParameters::DistanceType) const;

  /** initialiser les valeurs du centroide a partir d'une instance de BDD */
  void InitializeModelingCentroidValues(const KWObject *);

  /** initialiser les valeurs du centroide a partir d'une instance de cluster
   * KMean */
  void InitializeModelingCentroidValues(const KMClusterInstance *);

  /** obtenir la valeur de l'intertie intra, qui doit avoir été auparavant
   * calculée */
  const Continuous GetInertyIntra(KMParameters::DistanceType) const;

  /** obtenir, pour un attribut particulier,  la valeur de l'intertie intra, qui
   * doit avoir été auparavant calculée */
  const Continuous GetInertyIntraForAttribute(const int attributeRank,
                                              KMParameters::DistanceType) const;

  /** obtenir la valeur de l'intertie inter, qui doit avoir été auparavant
   * calculée */
  const Continuous GetInertyInter(KMParameters::DistanceType) const;

  /** nombre d'instances du cluster (permet de ne pas dépendre de GetCount(),
   * meme si on a vidé un cluster de ses instances, en ne conservant que ses
   * statistiques */
  const longint GetFrequency() const;

  /** "forcer" le nombre d'instances du cluster (a partir des valeurs lues dans
   * un dictionnaire, par exemple)  */
  void SetFrequency(const longint freq);

  /** index du cluster dans la liste de clusters */
  void SetIndex(const int);

  /** index du cluster dans la liste de clusters */
  int GetIndex() const;

  /** rapport entre le nombre d'instances du cluster et le nombre total
   * d'instances */
  const double GetCoverage(const longint totalInstancesNumber) const;

  /** compacite du cluster */
  const double GetCompactness() const;

  /** obtenir les probas des valeurs cibles (mode supervisé) */
  const ContinuousVector &GetTargetProbs() const;

  /** valeur moyenne des attributs natifs continus */
  Continuous
  GetNativeAttributeContinuousMeanValue(const KWAttribute *attr) const;

  /** valeur mediane des attributs natifs continus */
  Continuous
  GetNativeAttributeContinuousMedianValue(const KWAttribute *attr) const;

  /** nombre de valeurs manquantes pour un attribut donne */
  int GetMissingValues(const KWAttribute *attr) const;

  /** initialiser les probas des valeurs cibles (a partir des valeurs lues dans
   * un dictionnaire, par exemple) */
  void SetTargetProbs(const ContinuousVector &);

  /** repertorier quel est le cluster qui est le plus proche */
  void SetNearestCluster(KMCluster *);

  /** savoir quel est le cluster qui est le plus proche */
  KMCluster *GetNearestCluster() const;

  /** valeur majoritaire de la cible */
  const ALString &GetMajorityTargetValue() const;

  /** index de la valeur majoritaire de la cible */
  int GetMajorityTargetIndex() const;

  /** incrementer le nombre de valeurs manquantes pour les attributs natifs
   * d'une instance */
  void
  IncrementInstancesWithMissingNativeValuesNumber(const KWObject *instance);

  /** clonage du cluster, dans un nouveau cluster */
  KMCluster *Clone();

  /* remplacer les instances existantes, par les instances du cluster source */
  void CopyInstancesFrom(const KMCluster *source);

  /** copie des valeurs d'un cluster dans le cluster courant */
  void CopyFrom(const KMCluster *aSource);

  void Write(ostream &ost) const;

  /** savoir si les statistiques du cluster doivent etre recalculees, suite a la
   * modification de la liste de ses instances */
  const bool IsStatisticsUpToDate() const;

  /** indiquer si les statistiques du cluster doivent etre recalculees */
  void SetStatisticsUpToDate(const bool);

  ALString GetLabel() const;

  void SetLabel(const ALString &);

  /** distance d'un objet de BDD par rapport a un centroide passé en paramètre,
   * selon un type de calcul de distance donné */
  Continuous FindDistanceFromCentroid(const KWObject *o1,
                                      const ContinuousVector &centroids,
                                      KMParameters::DistanceType);

  /** distance de l'attribut d'un objet par rapport a un centroide passé en
   * paramètre, selon un type de calcul de distance donné */
  Continuous FindDistanceFromCentroid(const KWObject *o1,
                                      const ContinuousVector &centroids,
                                      KMParameters::DistanceType,
                                      const int attributeRankInCentroid);

  /** distance d'un objet "instance de cluster" par rapport a un centroide passé
   * en paramètre, selon un type de calcul de distance donné */
  Continuous FindDistanceFromCentroid(const KMClusterInstance *,
                                      const ContinuousVector &centroids,
                                      KMParameters::DistanceType);

  /** calculer les statistiques de fin d'iteration, sur l'ensemble des instances
   * du cluster, pendant un clustering */
  void ComputeIterationStatistics();

  /** mettre a jour la somme des distances des instances par rapport au centre
   * du cluster, sur l'ensemble des instances du cluster */
  void ComputeDistanceSum(KMParameters::DistanceType);

  /** mettre a jour les valeurs du centroide, en mode de calcul "moyenne", sur
   * l'ensemble des instances du cluster */
  void ComputeMeanModelingCentroidValues();

  /** mettre a jour les valeurs du centroide de modeling, en mode de calcul
   * "mediane", sur l'ensemble des instances du cluster */
  void ComputeMedianModelingCentroidValues();

  /** mettre a jour les valeurs du centroide d'evaluation, en mode de calcul
   * "mediane", sur l'ensemble des instances du cluster */
  void ComputeMedianEvaluationCentroidValues();

  /** determiner quelle est l'instance la plus proche du centre virtuel, sur
   * l'ensemble des instances du cluster */
  void ComputeInstanceNearestToCentroid(KMParameters::DistanceType);

  /** determiner quelle est l'instance la plus eloignee du centre virtuel, sur
   * l'ensemble des instances du cluster */
  void ComputeInstanceFurthestToCentroid(KMParameters::DistanceType);

  /** calcul de l'inertie intra du cluster, tous attributs confondus, sur
   * l'ensemble des instances du cluster */
  const Continuous ComputeInertyIntra(KMParameters::DistanceType);

  /** calcul de l'inertie intra du cluster, pour un attribut particulier, sur
   * l'ensemble des instances du cluster */
  const Continuous ComputeInertyIntraForAttribute(const int attributeRank,
                                                  KMParameters::DistanceType);

  /** calcul de la moyenne du cluster, pour un attribut KMean particulier, sur
   * l'ensemble des instances du cluster */
  const Continuous
  ComputeMeanValueForAttribute(const KWLoadIndex &attributeLoadIndex,
                               KMParameters::DistanceType);

  /** calcul de l'inertie inter */
  const Continuous
  ComputeInertyInter(KMParameters::DistanceType distanceType,
                     const ContinuousVector &globalCentroidValues,
                     const longint totalFrequency,
                     const boolean bUseEvaluationCentroids = false);

  /** calcul des moyennes, pour les attributs de type Continuous, sur l'ensemble
   * des instances du cluster */
  void ComputeNativeAttributesContinuousMeanValues();

  /** calcul des medianes, pour les attributs de type Continuous, sur l'ensemble
   * des instances du cluster */
  void ComputeNativeAttributesContinuousMedianValues();

  /** calcul de la repartition des valeurs reelles de l'attribut cible (mode
   * supervisé), sur l'ensemble des instances du cluster */
  void ComputeTrainingTargetProbs(const ObjectArray &targetAttributeValues,
                                  const KWAttribute *targetAttribute);

  /** classe cible majoritaire du cluster */
  void ComputeMajorityTargetValue(const ObjectArray &targetAttributeValues);

  /** mesure de la compacite d'un cluster */
  const Continuous ComputeCompactness(const ObjectArray &targetAttributeValues,
                                      const KWAttribute *targetAttribute);

  // mise a jour incrementale des stats, en fonction de l'affectation d'une
  // nouvelle instance au cluster. Sert lorsqu'on ne souhaite pas stocker les
  // instances dans les clusters (phase d'evaluation, apprentissage
  // mini-batch...)

  /** mise a jour incrementale de la somme des distances, en fonction de l'ajout
   * d'une nouvelle instance */
  void UpdateDistanceSum(KMParameters::DistanceType,
                         const KWObject *newInstance,
                         const ContinuousVector &cvCentroidValues);

  /** mise a jour incrementale des valeurs de controide, en fonction de l'ajout
   * d'une nouvelle instance  */
  void UpdateMeanCentroidValues(const KWObject *newInstance,
                                ContinuousVector &cvCentroidValues);

  /** mise a jour incrementale des moyennes des attributs continus, en fonction
   * de l'ajout d'une nouvelle instance  */
  void UpdateNativeAttributesContinuousMeanValues(const KWObject *);

  /** mise a jour incrementale de l'inertie intra cluster, en fonction de
   * l'ajout d'une nouvelle instance  */
  void UpdateInertyIntra(KMParameters::DistanceType,
                         const KWObject *newInstance,
                         const ContinuousVector &cvCentroidValues);

  /** mise a jour incrementale de la compacite du cluster, en fonction de
   * l'ajout d'une nouvelle instance  */
  void UpdateCompactness(const KWObject *newInstance,
                         const ObjectArray &targetAttributeValues,
                         const KWAttribute *targetAttribute,
                         const ContinuousVector &gravityCenter);

  /** mise a jour incrementale ds stats de l'attribut cible, en fonction de
   * l'ajout d'une nouvelle instance  */
  void UpdateTargetProbs(const ObjectArray &targetAttributeValues,
                         const KWAttribute *targetAttribute,
                         const KWObject *newInstance);

  /** evalue si l'instance passe en parametre est la plus proche du centre. Si
   * oui, retourne True et remplace l'ancienne instance la plus proche. Sinon,
   * retourne False  */
  boolean
  UpdateInstanceNearestToCentroid(KMParameters::DistanceType,
                                  const KWObject *newObject,
                                  const ContinuousVector &cvCentroidValues);

  /** mise a jour incrementale de l'inertie intra pour un attribut donne, en
   * fonction de l'ajout d'une nouvelle instance  */
  void UpdateInertyIntraForAttribute(const KWObject *newInstance,
                                     const int attributeLoadIndex,
                                     KMParameters::DistanceType);

  /** finalisation du calcul des stats incerementales (c'est a dire, calculees
   * instance par instance) */
  void FinalizeStatisticsUpdateFromInstances();

protected:
  // attributs

  /** parametrage du clustering */
  const KMParameters *parameters;

  /** indique si les stats calculées sont synchronisees avec les instances du
   * cluster */
  bool bStatisticsUpToDate;

  /** valeurs du centroide (centre virtuel) */
  ContinuousVector cvModelingCentroidValues;

  /** valeurs initiales du centroide, avant convergence (centre virtuel) */
  ContinuousVector cvInitialCentroidValues;

  /** valeurs du centroide d'un cluster, issu de l'evaluation (centre virtuel)
   */
  ContinuousVector cvEvaluationCentroidValues;

  /** somme des distances des instances par rapport au centre */
  ContinuousVector cvDistancesSum;

  /** inertie intra du cluster, tous attributs confondus (une valeur par type de
   * distance) */
  ContinuousVector cvInertyIntra;

  /** inertie intra du cluster, par attribut, pour la norme L1 (les postes
   * correspondent aux index de chargement des attributs) */
  ContinuousVector cvInertyIntraL1ByAttributes;

  /** inertie intra du cluster, par attribut, pour la norme L2 (les postes
   * correspondent aux index de chargement des attributs) */
  ContinuousVector cvInertyIntraL2ByAttributes;

  /** inertie intra du cluster, par attribut, pour la norme cosinus (les postes
   * correspondent aux index de chargement des attributs) */
  ContinuousVector cvInertyIntraCosineByAttributes;

  /** inertie inter du cluster, tous attributs confondus (une valeur par type de
   * distance) */
  ContinuousVector cvInertyInter;

  /** moyennes des attributs continuous du cluster */
  ContinuousVector cvNativeAttributesContinuousMeanValues;

  /** medianes des attributs continuous du cluster */
  ContinuousVector cvNativeAttributesContinuousMedianValues;

  /** probas des valeurs cibles reelles, lors d'une evaluation (mode supervisé)
   */
  ContinuousVector cvTargetProbs;

  /** chaque poste correspond a un rang attribut, et contient le nombre
   * d'attributs natifs ayant une valeur manquante */
  IntVector ivMissingNativeValues;

  /** nombre d'instances, mis à jour soit à partir d'un dictionnaire, soit apres
   * une iteration */
  longint lFrequency;

  /** couverture du cluster, par rapport au nombre total d'instances */
  double dCoverage;

  /** compacite du cluster */
  double dCompactness;

  // distance minimale d'une instance de ce cluster au centroide
  double dMinDistanceFromCentroid;

  /** instance reelle la plus proche du centroide du cluster */
  KMClusterInstance *instanceNearestToCentroid;

  /** instance reelle la plus eloignee du centroide du cluster */
  KMClusterInstance *instanceFurthestToCentroid;

  /** cluster le plus proche de ce cluster */
  KMCluster *nearestCluster;

  /** index du cluster dans la liste de clusters */
  int iIndex;

  /** valeur majoritaire de la cible */
  ALString sMajorityTargetValue;

  /** index de la valeur majoritaire de la cible */
  int iMajorityTargetIndex;

  ALString sLabel;

  friend class PLShared_Cluster;
};

//////////////////////////////////////////////////////////
// Classe PLShared_Cluster
/// Serialisation de la classe KMCluster

class PLShared_Cluster : public PLSharedObject {

public:
  PLShared_Cluster();
  ~PLShared_Cluster();

  // Acces au cluster
  void SetCluster(KMCluster *);
  KMCluster *GetCluster();

  // Reimplementation des methodes virtuelles
  void DeserializeObject(PLSerializer *, Object *) const;
  void SerializeObject(PLSerializer *, const Object *) const override;

  //////////////////////////////////////////////////////////////////
  ///// Implementation
protected:
  // Creation d'un objet (type d'objet a serialiser)
  Object *Create() const;

  // Test de la serialisation du cluster passe en parametre (celui-ci est
  // detruit a la fin du test)
  static boolean TestClusterSerialization(KMCluster *testCluster);
};

// implementations inline

inline const bool KMCluster::IsStatisticsUpToDate() const {
  return bStatisticsUpToDate;
}

inline const double KMCluster::GetCompactness() const { return dCompactness; }

inline void KMCluster::AddInstance(KWObject *o) {
  require(o != NULL);
  this->SetAt(o, o); // La valeur n'est nécessaire qu'afin de pouvoir faire un
                     // DeleteAll(), à la fin de l'évaluation (cf. methode
                     // KMClassifierEvaluation::Evaluate() )
  bStatisticsUpToDate = false;
}

inline KWObject *KMCluster::RemoveInstance(KWObject *o) {
  this->RemoveKey(o);
  bStatisticsUpToDate = false;
  return (o);
}

inline const KMClusterInstance *
KMCluster::GetInstanceNearestToCentroid() const {
  return instanceNearestToCentroid;
}

inline const KMClusterInstance *
KMCluster::GetInstanceFurthestToCentroid() const {
  return instanceFurthestToCentroid;
}

inline const ContinuousVector &KMCluster::GetModelingCentroidValues() const {
  return cvModelingCentroidValues;
}

inline const ContinuousVector &KMCluster::GetEvaluationCentroidValues() const {
  return cvEvaluationCentroidValues;
}

inline const ContinuousVector &KMCluster::GetInitialCentroidValues() const {
  return cvInitialCentroidValues;
}

inline const ALString KMCluster::GetClassLabel() const {
  return "Cluster K-Mean";
}

inline const KMParameters *KMCluster::GetParameters() const {
  return parameters;
}
inline void KMCluster::SetParameters(const KMParameters *p) { parameters = p; }

inline const Continuous
KMCluster::GetMeanDistance(KMParameters::DistanceType d) const {
  if (GetCount() == 0 or cvDistancesSum.GetAt(d) == 0)
    return 0;
  else
    return (cvDistancesSum.GetAt(d) / GetCount());
}

inline const Continuous
KMCluster::GetDistanceSum(KMParameters::DistanceType d) const {
  return cvDistancesSum.GetAt(d);
}
inline const Continuous
KMCluster::GetInertyIntra(KMParameters::DistanceType d) const {
  return cvInertyIntra.GetAt(d);
}

inline const Continuous
KMCluster::GetInertyIntraForAttribute(const int attributeRank,
                                      KMParameters::DistanceType d) const {

  assert(attributeRank >= 0);
  assert(d == KMParameters::L1Norm or d == KMParameters::L2Norm or
         d == KMParameters::CosineNorm);

  if (d == KMParameters::L1Norm)
    return cvInertyIntraL1ByAttributes.GetAt(attributeRank);
  else if (d == KMParameters::L2Norm)
    return cvInertyIntraL2ByAttributes.GetAt(attributeRank);
  else
    return cvInertyIntraCosineByAttributes.GetAt(attributeRank);
}

inline const Continuous
KMCluster::GetInertyInter(KMParameters::DistanceType d) const {
  return cvInertyInter.GetAt(d);
}
inline const longint KMCluster::GetFrequency() const { return lFrequency; }
inline void KMCluster::SetFrequency(const longint i) { lFrequency = i; }
inline const double
KMCluster::GetCoverage(const longint totalInstancesNumber) const {
  return (double)lFrequency / totalInstancesNumber;
}
inline int KMCluster::GetIndex() const { return iIndex; }
inline void KMCluster::SetIndex(const int i) { iIndex = i; }
inline const ContinuousVector &KMCluster::GetTargetProbs() const {

  return cvTargetProbs;
}

inline void KMCluster::SetNearestCluster(KMCluster *c) { nearestCluster = c; }

inline KMCluster *KMCluster::GetNearestCluster() const {
  return nearestCluster;
}

inline const ALString &KMCluster::GetMajorityTargetValue() const {
  return sMajorityTargetValue;
}

inline int KMCluster::GetMajorityTargetIndex() const {
  return iMajorityTargetIndex;
}
