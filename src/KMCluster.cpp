// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text
// of which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or
// see the "LICENSE" file for more details.

#include "KMCluster.h"
#include "KMClustering.h"

KMCluster::KMCluster(const KMParameters *params) : parameters(params) {
  InitializeStatistics();
  iIndex = -1;
}
KMCluster::~KMCluster(void) {
  if (instanceNearestToCentroid != NULL)
    delete instanceNearestToCentroid;

  if (instanceFurthestToCentroid != NULL)
    delete instanceFurthestToCentroid;
}

void KMCluster::InitializeStatistics() {

  bStatisticsUpToDate = true;
  cvDistancesSum.SetSize(3); // 3 normes a ce jour : L1, L2 et Cosinus
  cvDistancesSum.Initialize();
  cvInertyIntra.SetSize(3);
  cvInertyIntra.Initialize();
  cvInertyInter.SetSize(3);
  cvInertyInter.Initialize();
  lFrequency = 0;
  dCoverage = 0;
  dCompactness = 0;
  dMinDistanceFromCentroid = 0;
  instanceNearestToCentroid = NULL;
  instanceFurthestToCentroid = NULL;
  nearestCluster = NULL;
  iMajorityTargetIndex = -1;
}

void KMCluster::InitializeModelingCentroidValues(const KWObject *o) {

  // initialiser le centroide a partir des valeurs de l'objet de BDD recu en
  // parametre tous les attributs sont representes dans le vector, y compris les
  // attributs non KMean, pour des raisons de simplicité et performance

  require(o != NULL);

  const int nbAttr = o->GetClass()->GetLoadedAttributeNumber();
  require(nbAttr != 0);

  cvModelingCentroidValues.SetSize(nbAttr);
  cvModelingCentroidValues.Initialize();

  assert(parameters->GetKMeanAttributesLoadIndexes().GetSize() == nbAttr);

  for (int i = 0; i < nbAttr; i++) {

    const KWLoadIndex loadIndex =
        parameters->GetKMeanAttributesLoadIndexes().GetAt(i);

    if (loadIndex.IsValid())
      // il s'agit bien d'un attribut KMeans
      cvModelingCentroidValues.SetAt(i, o->GetContinuousValueAt(loadIndex));
  }
}

/** initialiser les valeurs du centroide a partir d'une instance de cluster
 * KMean */
void KMCluster::InitializeModelingCentroidValues(
    const KMClusterInstance *clusterInstance) {

  require(clusterInstance != NULL);

  const int nbAttr = clusterInstance->GetLoadedAttributes().GetSize();
  require(nbAttr != 0);

  cvModelingCentroidValues.SetSize(nbAttr);
  cvModelingCentroidValues.Initialize();

  assert(nbAttr == parameters->GetKMeanAttributesLoadIndexes().GetSize());

  for (int i = 0; i < nbAttr; i++) {
    const KWLoadIndex loadIndex =
        parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
    if (loadIndex.IsValid())
      // il s'agit bien d'un attribut KMeans
      cvModelingCentroidValues.SetAt(
          i, clusterInstance->GetContinuousValueAt(loadIndex));
  }
}

void KMCluster::SetModelingCentroidValues(
    const ContinuousVector &newCentroids) {

  cvModelingCentroidValues.CopyFrom(&newCentroids);
}

void KMCluster::SetInitialCentroidValues(const ContinuousVector &newCentroids) {

  cvInitialCentroidValues.CopyFrom(&newCentroids);
}

void KMCluster::ComputeInstanceNearestToCentroid(
    KMParameters::DistanceType distanceType) {
  if (GetCount() == 0) {
    AddWarning("Can't compute the nearest instance to centroid, on cluster " +
               sLabel + ", because it does not contain any element.");
    return;
  }

  double minimumDistance = 0.0;
  KWObject *nearestInstance = NULL;
  bool firstLoop = true;

  POSITION position = GetStartPosition();
  NUMERIC key;
  Object *oCurrent;

  while (position != NULL) {

    GetNextAssoc(position, key, oCurrent);
    KWObject *currentInstance = static_cast<KWObject *>(oCurrent);

    double distance = FindDistanceFromCentroid(
        currentInstance, cvModelingCentroidValues, distanceType);

    if (firstLoop) {
      firstLoop = false;
      minimumDistance = distance;
      nearestInstance = currentInstance;
    } else {
      if (minimumDistance > distance) {
        minimumDistance = distance;
        nearestInstance = currentInstance;
      }
    }
  }

  assert(nearestInstance != NULL);

  if (instanceNearestToCentroid != NULL)
    delete instanceNearestToCentroid;

  instanceNearestToCentroid =
      new KMClusterInstance(nearestInstance, parameters);
}
void KMCluster::ComputeInstanceFurthestToCentroid(
    KMParameters::DistanceType distanceType) {
  assert(GetCount() > 0); // cluster clone, qui possede des centroides mais pas
                          // d'instances. Dans ce cas, l'instance la plus proche
                          // du centroide est celle du cluster qui a ete clone

  // "blindage" pour mode release : : ne devrait jamais arriver, mais eviter le
  // crash, le cas echeant
  if (GetCount() == 0) {
    AddWarning("Can't compute the furthest instance to centroid, on cluster " +
               sLabel + ", because it does not contain any element.");
    return;
  }

  double maximumDistance = 0.0;
  KWObject *furthestInstance = NULL;
  bool firstLoop = true;

  POSITION position = GetStartPosition();
  NUMERIC key;
  Object *oCurrent;

  while (position != NULL) {

    GetNextAssoc(position, key, oCurrent);
    KWObject *currentInstance = static_cast<KWObject *>(oCurrent);

    double distance = FindDistanceFromCentroid(
        currentInstance, cvModelingCentroidValues, distanceType);

    if (firstLoop) {
      firstLoop = false;
      maximumDistance = distance;
      furthestInstance = currentInstance;
    } else {
      if (distance > maximumDistance) {
        maximumDistance = distance;
        furthestInstance = currentInstance;
      }
    }
  }

  assert(furthestInstance != NULL);

  if (instanceFurthestToCentroid != NULL)
    delete instanceFurthestToCentroid;

  instanceFurthestToCentroid =
      new KMClusterInstance(furthestInstance, parameters);
}

void KMCluster::ComputeIterationStatistics() {
  // NB. un cluster clone est considere comme etant a jour, du point de vue de
  // ses stats internes. Il ne faut pas recalculer ses stats, sinon elles seront
  // faussees, puisqu'il ne contient plus d'instances.

  if (bStatisticsUpToDate)
    return;

  lFrequency =
      GetCount(); // garder la memoire du nombre d'individus du cluster, meme en
                  // cas de clonage (qui ne garde pas les instances reelles)

  if (lFrequency == 0) { // le cluster a ete vidé suite a une itération
    // reinitialiser toutes les statistiques :
    cvDistancesSum.Initialize();
    cvModelingCentroidValues.Initialize();
  } else {

    // mise a jour des centroides
    ComputeMeanModelingCentroidValues();

    // mise a jour de la somme des distances
    ComputeDistanceSum(parameters->GetDistanceType());
  }

  bStatisticsUpToDate = true;
}

void KMCluster::ComputeMeanModelingCentroidValues() {

  if (GetCount() == 0) {
    AddWarning("Can't compute mean centroid values, on cluster " + sLabel +
               ", because it does not contain any element.");
    return;
  }

  POSITION position = GetStartPosition();
  NUMERIC key;
  Object *oCurrent;
  GetNextAssoc(position, key, oCurrent);
  KWObject *currentInstance = static_cast<KWObject *>(oCurrent);

  const int nbAttr = currentInstance->GetClass()->GetLoadedAttributeNumber();
  require(nbAttr != 0);

  if (cvModelingCentroidValues.GetSize() == 0) { // premiere maj
    cvModelingCentroidValues.SetSize(nbAttr);
    cvModelingCentroidValues.Initialize();
  }

  ContinuousVector sums;

  sums.SetSize(nbAttr);

  sums.Initialize();

  const int size = parameters->GetKMeanAttributesLoadIndexes().GetSize();

  // balayer toutes les instances pour mettre a jour les valeurs de centroides
  position = GetStartPosition();

  while (position != NULL) {

    GetNextAssoc(position, key, oCurrent);
    currentInstance = static_cast<KWObject *>(oCurrent);

    for (int i = 0; i < size; i++) {
      const KWLoadIndex loadIndex =
          parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
      if (loadIndex.IsValid())
        sums.SetAt(i, sums.GetAt(i) +
                          currentInstance->GetContinuousValueAt(loadIndex));
    }
  }

  for (int i = 0; i < size; i++) {
    cvModelingCentroidValues.SetAt(i, (sums.GetAt(i) / GetCount()));
  }
}

void KMCluster::ComputeMedianModelingCentroidValues() {

  if (GetCount() == 0) {
    AddWarning("Can't compute median centroid values, on cluster " + sLabel +
               ", because it does not contain any element.");
    return;
  }

  POSITION position = GetStartPosition();
  NUMERIC key;
  Object *oCurrent;
  GetNextAssoc(position, key, oCurrent);
  KWObject *currentInstance = static_cast<KWObject *>(oCurrent);

  const int nbAttr = currentInstance->GetClass()->GetLoadedAttributeNumber();
  require(nbAttr != 0);

  if (cvModelingCentroidValues.GetSize() == 0) { // premiere maj
    cvModelingCentroidValues.SetSize(nbAttr);
    cvModelingCentroidValues.Initialize();
  }

  assert(nbAttr == parameters->GetKMeanAttributesLoadIndexes().GetSize());

  for (int i = 0; i < nbAttr; i++) {

    const KWLoadIndex loadIndex =
        parameters->GetKMeanAttributesLoadIndexes().GetAt(i);

    if (not loadIndex.IsValid())
      // il ne s'agit pas d'un attribut KMeans
      continue;

    ContinuousVector cvValues;

    // balayer toutes les instances pour mettre a jour les valeurs de centroides
    position = GetStartPosition();

    while (position != NULL) {

      GetNextAssoc(position, key, oCurrent);
      currentInstance = static_cast<KWObject *>(oCurrent);
      cvValues.Add(currentInstance->GetContinuousValueAt(loadIndex));
    }

    // Calcul de la valeur mediane
    assert(cvValues.GetSize() > 0);

    if (cvValues.GetSize() == 1)
      cvModelingCentroidValues.SetAt(i, cvValues.GetAt(0));
    // Cas ou il y a au moins deux valeurs
    else {
      // Tri des valeurs
      cvValues.Sort();

      // Calcul de la valeur mediane, selon la parite de la taille du tableau de
      // valeurs
      if (cvValues.GetSize() % 2 == 0)
        cvModelingCentroidValues.SetAt(
            i, (cvValues.GetAt(cvValues.GetSize() / 2 - 1) +
                cvValues.GetAt(cvValues.GetSize() / 2)) /
                   2);
      else
        cvModelingCentroidValues.SetAt(i,
                                       cvValues.GetAt(cvValues.GetSize() / 2));
    }
  }
}

void KMCluster::ComputeMedianEvaluationCentroidValues() {

  if (GetCount() == 0) {
    AddWarning("Can't compute median evaluation centroid values, on cluster " +
               sLabel + ", because it does not contain any element.");
    return;
  }

  POSITION position = GetStartPosition();
  NUMERIC key;
  Object *oCurrent;
  GetNextAssoc(position, key, oCurrent);
  KWObject *currentInstance = static_cast<KWObject *>(oCurrent);

  const int nbAttr = currentInstance->GetClass()->GetLoadedAttributeNumber();
  require(nbAttr != 0);

  if (cvEvaluationCentroidValues.GetSize() == 0) { // premiere maj
    cvEvaluationCentroidValues.SetSize(nbAttr);
    cvEvaluationCentroidValues.Initialize();
  }

  assert(nbAttr == parameters->GetKMeanAttributesLoadIndexes().GetSize());

  for (int i = 0; i < nbAttr; i++) {

    const KWLoadIndex loadIndex =
        parameters->GetKMeanAttributesLoadIndexes().GetAt(i);

    if (not loadIndex.IsValid())
      // il ne s'agit pas d'un attribut KMeans
      continue;

    ContinuousVector cvValues;

    // balayer toutes les instances pour mettre a jour les valeurs de centroides
    position = GetStartPosition();

    while (position != NULL) {

      GetNextAssoc(position, key, oCurrent);
      currentInstance = static_cast<KWObject *>(oCurrent);
      cvValues.Add(currentInstance->GetContinuousValueAt(loadIndex));
    }

    // Calcul de la valeur mediane
    assert(cvValues.GetSize() > 0);

    if (cvValues.GetSize() == 1)
      cvEvaluationCentroidValues.SetAt(i, cvValues.GetAt(0));
    // Cas ou il y a au moins deux valeurs
    else {
      // Tri des valeurs
      cvValues.Sort();

      // Calcul de la valeur mediane, selon la parite de la taille du tableau de
      // valeurs
      if (cvValues.GetSize() % 2 == 0)
        cvEvaluationCentroidValues.SetAt(
            i, (cvValues.GetAt(cvValues.GetSize() / 2 - 1) +
                cvValues.GetAt(cvValues.GetSize() / 2)) /
                   2);
      else
        cvEvaluationCentroidValues.SetAt(
            i, cvValues.GetAt(cvValues.GetSize() / 2));
    }
  }
}

void KMCluster::ComputeDistanceSum(KMParameters::DistanceType distanceType) {

  if (GetCount() == 0) {
    AddWarning("Can't compute distance sum, on cluster " + sLabel +
               ", because it does not contain any element.");
    return;
  }

  Continuous sum = 0.0;

  NUMERIC key;
  Object *oCurrent;
  POSITION position = GetStartPosition();

  while (position != NULL) {

    GetNextAssoc(position, key, oCurrent);
    KWObject *currentInstance = static_cast<KWObject *>(oCurrent);
    sum += FindDistanceFromCentroid(currentInstance, cvModelingCentroidValues,
                                    distanceType);
  }

  cvDistancesSum.SetAt(distanceType, sum);
}

void KMCluster::CopyInstancesFrom(const KMCluster *source) {

  RemoveAll();

  NUMERIC key;
  Object *oCurrent;

  POSITION position = source->GetStartPosition();

  while (position != NULL) {
    source->GetNextAssoc(position, key, oCurrent);
    KWObject *object = static_cast<KWObject *>(oCurrent);

    if (object != NULL)
      AddInstance(object);
  }
}

const Continuous
KMCluster::ComputeInertyIntra(KMParameters::DistanceType distanceType) {

  if (GetCount() == 0) {
    AddWarning("Can't compute inerty intra, on cluster " + sLabel +
               ", because it does not contain any element.");
    return 0;
  }

  NUMERIC key;
  Object *oCurrent;

  Continuous sum = 0.0;

  POSITION position = GetStartPosition();

  while (position != NULL) {
    GetNextAssoc(position, key, oCurrent);
    KWObject *object = static_cast<KWObject *>(oCurrent);

    if (object != NULL) {

      sum += FindDistanceFromCentroid(object, GetModelingCentroidValues(),
                                      distanceType);
    }
  }
  sum = (sum / GetCount());

  cvInertyIntra.SetAt(distanceType, sum);

  return sum;
}

const Continuous KMCluster::ComputeInertyIntraForAttribute(
    const int attributeRank, KMParameters::DistanceType distanceType) {

  // ne sert que pour la methode d'initialisation de cluster "variance
  // partitioning" (cf.
  // KMClusteringInitializer::InitializeVariancePartitioningCentroids) et pour
  // le calcul d'indice Davies Bouldin par attribut

  if (GetCount() == 0) {
    AddWarning("Can't compute attribute inerty intra, on cluster " + sLabel +
               ", because it does not contain any element.");
    return 0;
  }

  assert(cvModelingCentroidValues.GetSize() > 0);

  if (cvInertyIntraL1ByAttributes.GetSize() == 0) {
    cvInertyIntraL1ByAttributes.SetSize(cvModelingCentroidValues.GetSize());
    cvInertyIntraL1ByAttributes.Initialize();
  }
  if (cvInertyIntraL2ByAttributes.GetSize() == 0) {
    cvInertyIntraL2ByAttributes.SetSize(cvModelingCentroidValues.GetSize());
    cvInertyIntraL2ByAttributes.Initialize();
  }
  if (cvInertyIntraCosineByAttributes.GetSize() == 0) {
    cvInertyIntraCosineByAttributes.SetSize(cvModelingCentroidValues.GetSize());
    cvInertyIntraCosineByAttributes.Initialize();
  }

  Continuous sum = 0.0;

  NUMERIC key;
  Object *oCurrent;

  POSITION position = GetStartPosition();

  while (position != NULL) {
    GetNextAssoc(position, key, oCurrent);
    KWObject *object = static_cast<KWObject *>(oCurrent);

    if (object != NULL) {
      sum += FindDistanceFromCentroid(object, cvModelingCentroidValues,
                                      distanceType, attributeRank);
    }
  }
  sum = (sum / GetCount());

  if (distanceType == KMParameters::L1Norm)
    cvInertyIntraL1ByAttributes.SetAt(attributeRank, sum);
  else if (distanceType == KMParameters::L2Norm)
    cvInertyIntraL2ByAttributes.SetAt(attributeRank, sum);
  else
    cvInertyIntraCosineByAttributes.SetAt(attributeRank, sum);

  return sum;
}

void KMCluster::UpdateInertyIntraForAttribute(
    const KWObject *kwo, const int attributeRank,
    KMParameters::DistanceType distanceType) {

  assert(cvModelingCentroidValues.GetSize() > 0);

  if (cvInertyIntraL1ByAttributes.GetSize() == 0) {
    cvInertyIntraL1ByAttributes.SetSize(cvModelingCentroidValues.GetSize());
    cvInertyIntraL1ByAttributes.Initialize();
  }
  if (cvInertyIntraL2ByAttributes.GetSize() == 0) {
    cvInertyIntraL2ByAttributes.SetSize(cvModelingCentroidValues.GetSize());
    cvInertyIntraL2ByAttributes.Initialize();
  }
  if (cvInertyIntraCosineByAttributes.GetSize() == 0) {
    cvInertyIntraCosineByAttributes.SetSize(cvModelingCentroidValues.GetSize());
    cvInertyIntraCosineByAttributes.Initialize();
  }
  Continuous distance = FindDistanceFromCentroid(kwo, cvModelingCentroidValues,
                                                 distanceType, attributeRank);

  if (distanceType == KMParameters::L1Norm)
    cvInertyIntraL1ByAttributes.SetAt(
        attributeRank,
        cvInertyIntraL1ByAttributes.GetAt(attributeRank) + distance);
  else if (distanceType == KMParameters::L2Norm)
    cvInertyIntraL2ByAttributes.SetAt(
        attributeRank,
        cvInertyIntraL2ByAttributes.GetAt(attributeRank) + distance);
  else
    cvInertyIntraCosineByAttributes.SetAt(
        attributeRank,
        cvInertyIntraCosineByAttributes.GetAt(attributeRank) + distance);
}

const Continuous KMCluster::ComputeMeanValueForAttribute(
    const KWLoadIndex &attributeLoadIndex,
    KMParameters::DistanceType distanceType) {

  if (GetCount() == 0) {
    AddWarning("Can't compute attribute mean value, on cluster " + sLabel +
               ", because it does not contain any element.");
    return 0;
  }

  Continuous mean = 0.0;

  NUMERIC key;
  Object *oCurrent;

  POSITION position = GetStartPosition();

  while (position != NULL) {

    GetNextAssoc(position, key, oCurrent);
    KWObject *object = static_cast<KWObject *>(oCurrent);

    if (object != NULL) {

      assert(object->GetClass()
                 ->GetAttributeAtLoadIndex(attributeLoadIndex)
                 ->GetType() == KWType::Continuous);

      mean += object->GetContinuousValueAt(attributeLoadIndex);
    }
  }

  mean = (mean / GetCount());

  return mean;
}

const Continuous
KMCluster::ComputeInertyInter(KMParameters::DistanceType distanceType,
                              const ContinuousVector &globalCentroidValues,
                              const longint totalFrequency,
                              const boolean bUseEvaluationCentroids) {

  Continuous result = 0.0;

  const ContinuousVector &clusterCentroidValues =
      (bUseEvaluationCentroids ? cvEvaluationCentroidValues
                               : cvModelingCentroidValues);

  if (clusterCentroidValues.GetSize() != globalCentroidValues.GetSize()) {
    // cluster devenu vide
    cvInertyInter.SetAt(distanceType, 0);
    return 0;
  }

  // variables pour calcul en norme cosinus
  Continuous numeratorCosinus = 0.0;
  Continuous denominatorInstanceCosinus = 0.0;
  Continuous denominatorCentroidCosinus = 0.0;

  for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize();
       i++) {

    const KWLoadIndex &loadIndex =
        parameters->GetKMeanAttributesLoadIndexes().GetAt(i);

    if (not loadIndex.IsValid())
      // pas un attribut KMean
      continue;

    if (distanceType == KMParameters::L2Norm) {
      const Continuous d =
          clusterCentroidValues.GetAt(i) - globalCentroidValues.GetAt(i);
      result += (d * d);
    } else {
      if (distanceType == KMParameters::L1Norm) {
        result += fabs(clusterCentroidValues.GetAt(i) -
                       globalCentroidValues.GetAt(i));
      } else {
        // norme cosinus
        numeratorCosinus +=
            clusterCentroidValues.GetAt(i) * globalCentroidValues.GetAt(i);
        denominatorInstanceCosinus += pow(clusterCentroidValues.GetAt(i), 2);
        denominatorCentroidCosinus += pow(globalCentroidValues.GetAt(i), 2);
      }
    }
  }

  if (distanceType == KMParameters::CosineNorm) {
    Continuous denominator =
        sqrt(denominatorInstanceCosinus) * sqrt(denominatorCentroidCosinus);
    result = 1 - (denominator == 0 ? 0 : numeratorCosinus / denominator);
  }

  result = (result / totalFrequency) * lFrequency;

  cvInertyInter.SetAt(distanceType, result);

  return result;
}

void KMCluster::ComputeTrainingTargetProbs(
    const ObjectArray &targetAttributeValues,
    const KWAttribute *targetAttribute) {

  require(targetAttributeValues.GetSize() > 0);
  require(targetAttribute != NULL);

  if (GetCount() == 0) {
    AddWarning("Can't compute training target probs, on cluster " + sLabel +
               ", because it does not contain any element.");
    return;
  }
  const KWLoadIndex &targetIndex = targetAttribute->GetLoadIndex();

  cvTargetProbs.SetSize(targetAttributeValues.GetSize());
  cvTargetProbs.Initialize();

  NUMERIC key;
  Object *oCurrent;

  POSITION position = GetStartPosition();

  while (position != NULL) {

    GetNextAssoc(position, key, oCurrent);
    KWObject *currentInstance = static_cast<KWObject *>(oCurrent);

    if (currentInstance != NULL) {

      ALString value =
          currentInstance->GetSymbolValueAt(targetIndex).GetValue();

      // rechercher l'index correspondant a la valeur de l'attribut, pour
      // renseigner notre tableau d'occurences
      int idx = 0;
      for (; idx < targetAttributeValues.GetSize(); idx++) {
        StringObject *s =
            cast(StringObject *, targetAttributeValues.GetAt(idx));
        if (value == s->GetString())
          break;
      }

      assert(idx !=
             targetAttributeValues
                 .GetSize()); // comme on est en apprentissage, la valeur cible
                              // doit forcement etre deja repertoriee

      // incrementer de 1 le nombre d'occurences pour cette valeur cible
      cvTargetProbs.SetAt(idx, cvTargetProbs.GetAt(idx) + 1);
    }
  }

  // transformer les nombres d'occurences calculés, en probas comprises entre 0
  // et 1
  for (int i = 0; i < cvTargetProbs.GetSize(); i++) {
    cvTargetProbs.SetAt(i, cvTargetProbs.GetAt(i) / GetCount());
  }

  ComputeMajorityTargetValue(targetAttributeValues);
}

void KMCluster::ComputeMajorityTargetValue(
    const ObjectArray &targetAttributeValues) {

  assert(targetAttributeValues.GetSize() > 0);

  if (cvTargetProbs.GetSize() == 0)
    // cluster devenu vide en evaluation ?
    return;

  // rechercher l'index de la proba cible la plus forte
  iMajorityTargetIndex = 0;
  Continuous bestProba = 0;

  for (int i = 0; i < cvTargetProbs.GetSize(); i++) {
    if (cvTargetProbs.GetAt(i) > bestProba) {
      bestProba = cvTargetProbs.GetAt(i);
      iMajorityTargetIndex = i;
    }
  }

  assert(iMajorityTargetIndex < targetAttributeValues.GetSize());

  StringObject *s =
      cast(StringObject *, targetAttributeValues.GetAt(iMajorityTargetIndex));
  sMajorityTargetValue = s->GetString();
}

const Continuous
KMCluster::ComputeCompactness(const ObjectArray &targetAttributeValues,
                              const KWAttribute *targetAttribute) {

  if (GetCount() == 0) {
    AddWarning("Can't compute compactness, on cluster " + sLabel +
               ", because it does not contain any element.");
    return 0;
  }
  POSITION position = GetStartPosition();
  NUMERIC key;
  Object *oCurrent;
  GetNextAssoc(position, key, oCurrent);
  KWObject *currentInstance = static_cast<KWObject *>(oCurrent);;

  const int nbAttr = currentInstance->GetClass()->GetLoadedAttributeNumber();
  require(nbAttr != 0);

  require(GetMajorityTargetIndex() != -1);

  ContinuousVector currentInstanceValues;
  currentInstanceValues.SetSize(nbAttr);
  currentInstanceValues.Initialize();

  dCompactness = 0;
  const ContinuousVector &gravityCenter = GetModelingCentroidValues();
  const ALString &majorityTarget = GetMajorityTargetValue();

  if (majorityTarget == "")
    return dCompactness; // cluster devenu vide

  const KWLoadIndex &targetIndex = targetAttribute->GetLoadIndex();

  position = GetStartPosition();

  while (position != NULL) {

    GetNextAssoc(position, key, oCurrent);
    currentInstance = static_cast<KWObject *>(oCurrent);

    ALString currentInstanceTargetValue =
        currentInstance->GetSymbolValueAt(targetIndex).GetValue();

    // representer l'instance sous forme de tableau de continus
    for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize();
         i++) {
      const KWLoadIndex &loadIndex =
          parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
      if (not loadIndex.IsValid())
        // pas un attribut KMean
        continue;
      currentInstanceValues.SetAt(
          i, currentInstance->GetContinuousValueAt(loadIndex));
    }

    dCompactness += KMClustering::GetSimilarityBetween(
        gravityCenter, currentInstanceValues, majorityTarget,
        currentInstanceTargetValue, parameters);
  }

  dCompactness = dCompactness / (double)GetCount();

  return dCompactness;
}

void KMCluster::UpdateCompactness(const KWObject *instance,
                                  const ObjectArray &targetAttributeValues,
                                  const KWAttribute *targetAttribute,
                                  const ContinuousVector &gravityCenter) {

  const int nbAttr = instance->GetClass()->GetLoadedAttributeNumber();
  require(nbAttr != 0);
  require(GetMajorityTargetIndex() != -1);
  assert(targetAttribute != NULL);

  ContinuousVector currentInstanceValues;
  currentInstanceValues.SetSize(nbAttr);
  currentInstanceValues.Initialize();

  const ALString &majorityTarget = GetMajorityTargetValue();

  if (majorityTarget == "")
    return; // cluster devenu vide

  const KWLoadIndex targetIndex = targetAttribute->GetLoadIndex();

  ALString currentInstanceTargetValue =
      instance->GetSymbolValueAt(targetIndex).GetValue();

  // representer l'instance sous forme de tableau de continus
  for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize();
       i++) {
    const KWLoadIndex loadIndex =
        parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
    if (not loadIndex.IsValid())
      // pas un attribut KMean
      continue;
    currentInstanceValues.SetAt(i, instance->GetContinuousValueAt(loadIndex));
  }

  dCompactness += KMClustering::GetSimilarityBetween(
      gravityCenter, currentInstanceValues, majorityTarget,
      currentInstanceTargetValue, parameters);
}

void KMCluster::UpdateDistanceSum(KMParameters::DistanceType distanceType,
                                  const KWObject *instance,
                                  const ContinuousVector &cvCentroidValues) {

  assert(cvCentroidValues.GetSize() > 0);

  Continuous c =
      FindDistanceFromCentroid(instance, cvCentroidValues, distanceType);

  cvDistancesSum.SetAt(distanceType, cvDistancesSum.GetAt(distanceType) + c);
}

void KMCluster::UpdateMeanCentroidValues(const KWObject *instance,
                                         ContinuousVector &cvCentroidValues) {

  const int nbAttr = instance->GetClass()->GetLoadedAttributeNumber();
  require(nbAttr != 0);

  if (cvCentroidValues.GetSize() == 0) { // premiere maj
    cvCentroidValues.SetSize(nbAttr);
    cvCentroidValues.Initialize();
  }

  assert(nbAttr == parameters->GetKMeanAttributesLoadIndexes().GetSize());

  for (int i = 0; i < nbAttr; i++) {
    const KWLoadIndex loadIndex =
        parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
    if (not loadIndex.IsValid())
      // pas un attribut KMean
      continue;
    Continuous c = (cvCentroidValues.GetAt(i) * (lFrequency - 1) +
                    instance->GetContinuousValueAt(loadIndex)) /
                   lFrequency;
    cvCentroidValues.SetAt(i, c);
  }
}

void KMCluster::UpdateNativeAttributesContinuousMeanValues(
    const KWObject *instance) {

  assert(instance != NULL);

  const KWLoadIndexVector &nativeAttributesLoadIndexes =
      parameters->GetNativeAttributesLoadIndexes();
  const int nbNativeAttributes = nativeAttributesLoadIndexes.GetSize();

  const KWClass *kwc = instance->GetClass();

  if (cvNativeAttributesContinuousMeanValues.GetSize() == 0) { // premiere maj
    cvNativeAttributesContinuousMeanValues.SetSize(
        kwc->GetLoadedAttributeNumber());
    cvNativeAttributesContinuousMeanValues.Initialize();
  }

  for (int idxNative = 0; idxNative < nbNativeAttributes; idxNative++) {

    const KWLoadIndex loadIndex = nativeAttributesLoadIndexes.GetAt(idxNative);
    if (not loadIndex.IsValid())
      continue;

    KWAttribute *native =
        instance->GetClass()->GetAttributeAtLoadIndex(loadIndex);
    assert(native != NULL);

    if (native->GetType() == KWType::Continuous and
        instance->GetContinuousValueAt(loadIndex) !=
            KWContinuous::GetMissingValue()) {

      cvNativeAttributesContinuousMeanValues.SetAt(
          idxNative, cvNativeAttributesContinuousMeanValues.GetAt(idxNative) +
                         instance->GetContinuousValueAt(loadIndex));
    }
  }
}

void KMCluster::UpdateInertyIntra(KMParameters::DistanceType distanceType,
                                  const KWObject *instance,
                                  const ContinuousVector &cvCentroidValues) {

  assert(instance != NULL);
  assert(cvCentroidValues.GetSize() > 0);

  Continuous c =
      FindDistanceFromCentroid(instance, cvCentroidValues, distanceType);

  cvInertyIntra.SetAt(
      distanceType,
      cvInertyIntra.GetAt(distanceType) +
          c); // lors de la finalisation, on divisera par la frequence
}

void KMCluster::UpdateTargetProbs(const ObjectArray &targetAttributeValues,
                                  const KWAttribute *targetAttribute,
                                  const KWObject *instance) {

  require(targetAttributeValues.GetSize() > 0);
  require(targetAttribute != NULL);
  require(instance != NULL);

  if (lFrequency == 1) {
    cvTargetProbs.SetSize(targetAttributeValues.GetSize());
    cvTargetProbs.Initialize();
  }

  ALString value =
      instance->GetSymbolValueAt(targetAttribute->GetLoadIndex()).GetValue();

  // rechercher l'index correspondant a la valeur de l'attribut, pour renseigner
  // notre tableau d'occurences
  int idx = 0;
  for (; idx < targetAttributeValues.GetSize(); idx++) {
    StringObject *s = cast(StringObject *, targetAttributeValues.GetAt(idx));
    if (value == s->GetString())
      break;
  }

  assert(idx < targetAttributeValues.GetSize());

  // incrementer de 1 le nombre d'occurences pour cette valeur cible
  cvTargetProbs.SetAt(idx, cvTargetProbs.GetAt(idx) + 1);
}

boolean KMCluster::UpdateInstanceNearestToCentroid(
    KMParameters::DistanceType distanceType, const KWObject *instance,
    const ContinuousVector &cvCentroidValues) {

  require(instance != NULL);

  if (instanceNearestToCentroid == NULL) {
    instanceNearestToCentroid = new KMClusterInstance(instance, parameters);
    dMinDistanceFromCentroid =
        FindDistanceFromCentroid(instance, cvCentroidValues, distanceType);
    return true;
  } else {

    const double distance =
        FindDistanceFromCentroid(instance, cvCentroidValues, distanceType);
    if (distance < dMinDistanceFromCentroid) {
      delete instanceNearestToCentroid;
      instanceNearestToCentroid = new KMClusterInstance(instance, parameters);
      dMinDistanceFromCentroid = distance;
      return true;
    }
  }
  return false;
}

Continuous
KMCluster::FindDistanceFromCentroid(const KWObject *o1,
                                    const ContinuousVector &centroids,
                                    KMParameters::DistanceType distanceType) {
  assert(centroids.GetSize() > 0);
  assert(o1 != NULL);

  Continuous result = 0.0;
  const int size = parameters->GetKMeanAttributesLoadIndexes().GetSize();

  if (distanceType == KMParameters::L2Norm) {

    for (int i = 0; i < size; i++) {
      const KWLoadIndex loadIndex =
          parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
      if (not loadIndex.IsValid())
        continue;
      const Continuous d =
          centroids.GetAt(i) - o1->GetContinuousValueAt(loadIndex);
      result += (d * d);
    }
  } else {
    if (distanceType == KMParameters::L1Norm) {
      for (int i = 0; i < size; i++) {
        const KWLoadIndex loadIndex =
            parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
        if (not loadIndex.IsValid())
          continue;
        result +=
            fabs(centroids.GetAt(i) - o1->GetContinuousValueAt(loadIndex));
      }
    } else {
      if (distanceType == KMParameters::CosineNorm) {
        Continuous numerator = 0.0;
        Continuous denominatorInstance = 0.0;
        Continuous denominatorCentroid = 0.0;

        for (int i = 0; i < size; i++) {
          const KWLoadIndex loadIndex =
              parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
          if (not loadIndex.IsValid())
            continue;
          numerator += centroids.GetAt(i) * o1->GetContinuousValueAt(loadIndex);
          denominatorInstance += pow(o1->GetContinuousValueAt(loadIndex), 2);
          denominatorCentroid += pow(centroids.GetAt(i), 2);
        }
        Continuous denominator =
            sqrt(denominatorInstance) * sqrt(denominatorCentroid);
        result = 1 - (denominator == 0 ? 0 : numerator / denominator);
      }
    }
  }

  return result;
}

Continuous KMCluster::FindDistanceFromCentroid(
    const KWObject *o1, const ContinuousVector &centroids,
    KMParameters::DistanceType distanceType, const int attributeRank) {
  assert(centroids.GetSize() > 0);
  assert(o1 != NULL);

  Continuous result = 0.0;

  if (distanceType == KMParameters::L2Norm) {
    KWLoadIndex attributeLoadIndex =
        parameters->GetKMeanAttributesLoadIndexes().GetAt(attributeRank);
    assert(attributeLoadIndex.IsValid());
    const Continuous d = centroids.GetAt(attributeRank) -
                         o1->GetContinuousValueAt(attributeLoadIndex);
    result = (d * d);
  } else {
    if (distanceType == KMParameters::L1Norm) {
      KWLoadIndex attributeLoadIndex =
          parameters->GetKMeanAttributesLoadIndexes().GetAt(attributeRank);
      assert(attributeLoadIndex.IsValid());
      result = fabs(centroids.GetAt(attributeRank) -
                    o1->GetContinuousValueAt(attributeLoadIndex));
    } else {
      if (distanceType == KMParameters::CosineNorm) {
        KWLoadIndex attributeLoadIndex =
            parameters->GetKMeanAttributesLoadIndexes().GetAt(attributeRank);
        assert(attributeLoadIndex.IsValid());
        Continuous numerator = centroids.GetAt(attributeRank) *
                               o1->GetContinuousValueAt(attributeLoadIndex);
        Continuous denominatorInstance =
            pow(o1->GetContinuousValueAt(attributeLoadIndex), 2);
        Continuous denominatorCentroid = pow(centroids.GetAt(attributeRank), 2);
        Continuous denominator =
            sqrt(denominatorInstance) * sqrt(denominatorCentroid);
        result = 1 - (denominator == 0 ? 0 : numerator / denominator);
      }
    }
  }

  return result;
}

Continuous
KMCluster::FindDistanceFromCentroid(const KMClusterInstance *clusterInstance,
                                    const ContinuousVector &centroids,
                                    KMParameters::DistanceType distanceType) {
  assert(centroids.GetSize() > 0);
  assert(clusterInstance != NULL);

  Continuous result = 0.0;
  const int size = parameters->GetKMeanAttributesLoadIndexes().GetSize();

  if (distanceType == KMParameters::L2Norm) {

    for (int i = 0; i < size; i++) {
      KWLoadIndex loadIndex =
          parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
      if (not loadIndex.IsValid())
        continue;
      const Continuous d =
          centroids.GetAt(i) - clusterInstance->GetContinuousValueAt(loadIndex);
      result += (d * d);
    }
  } else {
    if (distanceType == KMParameters::L1Norm) {
      for (int i = 0; i < size; i++) {
        KWLoadIndex loadIndex =
            parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
        if (not loadIndex.IsValid())
          continue;
        result += fabs(centroids.GetAt(i) -
                       clusterInstance->GetContinuousValueAt(loadIndex));
      }
    } else {
      if (distanceType == KMParameters::CosineNorm) {
        Continuous numerator = 0.0;
        Continuous denominatorInstance = 0.0;
        Continuous denominatorCentroid = 0.0;

        for (int i = 0; i < size; i++) {
          KWLoadIndex loadIndex =
              parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
          if (not loadIndex.IsValid())
            continue;
          numerator += centroids.GetAt(i) *
                       clusterInstance->GetContinuousValueAt(loadIndex);
          denominatorInstance +=
              pow(clusterInstance->GetContinuousValueAt(loadIndex), 2);
          denominatorCentroid += pow(centroids.GetAt(i), 2);
        }
        Continuous denominator =
            sqrt(denominatorInstance) * sqrt(denominatorCentroid);
        result = 1 - (denominator == 0 ? 0 : numerator / denominator);
      }
    }
  }

  return result;
}

void KMCluster::FinalizeStatisticsUpdateFromInstances() {

  // finalisation du calcul des stats "a la volee" (c'est a dire, calculees
  // instance par instance)

  // moyennes des attributs natifs continus : diviser les valeurs cumulees par
  // le nombre d'instances du cluster qui n'ont pas de valeurs manquantes pour
  // la valeur en question
  for (int i = 0; i < cvNativeAttributesContinuousMeanValues.GetSize(); i++) {

    if (cvNativeAttributesContinuousMeanValues.GetAt(i) == 0)
      continue; // ce n'est pas un attribut natif

    int missingValues = ivMissingNativeValues.GetSize() == 0
                            ? 0
                            : ivMissingNativeValues.GetAt(i);

    if (lFrequency - missingValues > 0)
      cvNativeAttributesContinuousMeanValues.SetAt(
          i, cvNativeAttributesContinuousMeanValues.GetAt(i) /
                 (lFrequency - missingValues));
    else
      cvNativeAttributesContinuousMeanValues.SetAt(i, 0);
  }

  // finalisation inerties intra
  for (int i = 0; i < cvInertyIntra.GetSize(); i++)
    cvInertyIntra.SetAt(i, cvInertyIntra.GetAt(i) / lFrequency);

  // diviser les inerties intra par attributs, par la frequence du cluster
  for (int i = 0; i < cvInertyIntraL1ByAttributes.GetSize(); i++) {
    cvInertyIntraL1ByAttributes.SetAt(i, cvInertyIntraL1ByAttributes.GetAt(i) /
                                             lFrequency);
  }
  for (int i = 0; i < cvInertyIntraL2ByAttributes.GetSize(); i++) {
    cvInertyIntraL2ByAttributes.SetAt(i, cvInertyIntraL2ByAttributes.GetAt(i) /
                                             lFrequency);
  }
  for (int i = 0; i < cvInertyIntraCosineByAttributes.GetSize(); i++) {
    cvInertyIntraCosineByAttributes.SetAt(
        i, cvInertyIntraCosineByAttributes.GetAt(i) / lFrequency);
  }

  // probas : transformer les nombres d'occurences calculés, en probas comprises
  // entre 0 et 1
  for (int i = 0; i < cvTargetProbs.GetSize(); i++)
    cvTargetProbs.SetAt(i, cvTargetProbs.GetAt(i) / lFrequency);

  // finalisation du calcul de compacite des clusters
  if (lFrequency > 0)
    dCompactness /= lFrequency;
  else
    dCompactness = 0;
}

KMCluster *KMCluster::Clone() {
  require(bStatisticsUpToDate); // ne pas creer un clone dont les stats n'ont
                                // pas été calculées ou "rafraichies"

  KMCluster *aClone = new KMCluster(parameters);
  aClone->CopyFrom(this);
  return aClone;
}

void KMCluster::CopyFrom(const KMCluster *aSource) {
  // copie a peu pres tout, sauf les instances elles-mêmes

  require(aSource != NULL);
  require(aSource->bStatisticsUpToDate);

  parameters = aSource->parameters;
  bStatisticsUpToDate = aSource->bStatisticsUpToDate;
  cvModelingCentroidValues.CopyFrom(&aSource->cvModelingCentroidValues);
  cvEvaluationCentroidValues.CopyFrom(&aSource->cvEvaluationCentroidValues);
  cvInitialCentroidValues.CopyFrom(&aSource->cvInitialCentroidValues);
  cvNativeAttributesContinuousMeanValues.CopyFrom(
      &aSource->cvNativeAttributesContinuousMeanValues);
  cvNativeAttributesContinuousMedianValues.CopyFrom(
      &aSource->cvNativeAttributesContinuousMedianValues);
  cvDistancesSum.CopyFrom(&aSource->cvDistancesSum);
  cvInertyIntra.CopyFrom(&aSource->cvInertyIntra);
  cvInertyIntraL1ByAttributes.CopyFrom(&aSource->cvInertyIntraL1ByAttributes);
  cvInertyIntraL2ByAttributes.CopyFrom(&aSource->cvInertyIntraL2ByAttributes);
  cvInertyIntraCosineByAttributes.CopyFrom(
      &aSource->cvInertyIntraCosineByAttributes);
  cvInertyInter.CopyFrom(&aSource->cvInertyInter);
  cvTargetProbs.CopyFrom(&aSource->cvTargetProbs);
  lFrequency = aSource->lFrequency;
  dCoverage = aSource->dCoverage;
  dMinDistanceFromCentroid = aSource->dMinDistanceFromCentroid;
  dCompactness = aSource->dCompactness;
  sLabel = aSource->sLabel;
  sMajorityTargetValue = aSource->sMajorityTargetValue;
  iMajorityTargetIndex = aSource->iMajorityTargetIndex;

  if (instanceNearestToCentroid != NULL)
    delete instanceNearestToCentroid;
  if (aSource->instanceNearestToCentroid != NULL)
    instanceNearestToCentroid = aSource->instanceNearestToCentroid->Clone();
  else
    instanceNearestToCentroid = NULL;

  if (instanceFurthestToCentroid != NULL)
    delete instanceFurthestToCentroid;
  if (aSource->instanceFurthestToCentroid != NULL)
    instanceFurthestToCentroid = aSource->instanceFurthestToCentroid->Clone();
  else
    instanceFurthestToCentroid = NULL;

  nearestCluster = aSource->nearestCluster;

  RemoveAll(); // enleve les instances eventuellement existantes
}

void KMCluster::SetTargetProbs(const ContinuousVector &source) {
  cvTargetProbs.CopyFrom(&source);
}

void KMCluster::ComputeNativeAttributesContinuousMeanValues() {

  if (GetCount() == 0) {
    AddWarning("Can't compute native attributes mean values, on cluster " +
               sLabel + ", because it does not contain any element.");
    return;
  }

  const KWLoadIndexVector &nativeAttributesLoadIndexes =
      parameters->GetNativeAttributesLoadIndexes();
  const int nbNativeAttributes = nativeAttributesLoadIndexes.GetSize();

  const KWClass *kwc = NULL;

  // parcourir toutes les instances du cluster
  NUMERIC key;
  Object *oCurrent;
  POSITION position = GetStartPosition();

  while (position != NULL) {

    GetNextAssoc(position, key, oCurrent);
    KWObject *instance = static_cast<KWObject *>(oCurrent);

    if (instance != NULL) {

      if (kwc == NULL) {
        kwc = instance->GetClass();
        cvNativeAttributesContinuousMeanValues.SetSize(
            kwc->GetLoadedAttributeNumber());
        cvNativeAttributesContinuousMeanValues.Initialize();
      }

      for (int i = 0; i < nbNativeAttributes; i++) {

        const KWLoadIndex loadIndex = nativeAttributesLoadIndexes.GetAt(i);
        if (not(loadIndex.IsValid()))
          continue;

        KWAttribute *native =
            instance->GetClass()->GetAttributeAtLoadIndex(loadIndex);
        assert(native != NULL);

        if (native->GetType() == KWType::Continuous and
            instance->GetContinuousValueAt(loadIndex) !=
                KWContinuous::GetMissingValue()) {
          cvNativeAttributesContinuousMeanValues.SetAt(
              i, cvNativeAttributesContinuousMeanValues.GetAt(i) +
                     instance->GetContinuousValueAt(loadIndex));
        }
      }
    }
  }

  assert(kwc != NULL);

  // diviser les valeurs cumulees par le nombre d'instances du cluster qui n'ont
  // pas de valeurs manquantes pour la valeur en question
  for (int i = 0; i < cvNativeAttributesContinuousMeanValues.GetSize(); i++) {

    if (cvNativeAttributesContinuousMeanValues.GetAt(i) == 0)
      continue; // ce n'est pas un attribut natif

    int missingValues = ivMissingNativeValues.GetSize() == 0
                            ? 0
                            : ivMissingNativeValues.GetAt(i);

    if (GetCount() - missingValues > 0)
      cvNativeAttributesContinuousMeanValues.SetAt(
          i, cvNativeAttributesContinuousMeanValues.GetAt(i) /
                 (GetCount() - missingValues));
    else
      cvNativeAttributesContinuousMeanValues.SetAt(i, 0);
  }
}

void KMCluster::ComputeNativeAttributesContinuousMedianValues() {

  if (GetCount() == 0) {
    AddWarning("Can't compute native attributes continuous median values, on "
               "cluster " +
               sLabel + ", because it does not contain any element.");
    return;
  }

  const KWClass *kwc = NULL;
  ObjectArray *continuousValues = NULL;
  int idxInstance = 0;

  const KWLoadIndexVector &nativeAttributesLoadIndexes =
      parameters->GetNativeAttributesLoadIndexes();
  const int nbNativeAttributes = nativeAttributesLoadIndexes.GetSize();

  // parcourir toutes les instances du cluster

  NUMERIC key;
  Object *oCurrent;
  POSITION position = GetStartPosition();

  while (position != NULL) {

    GetNextAssoc(position, key, oCurrent);
    KWObject *instance = static_cast<KWObject *>(oCurrent);

    if (instance != NULL) {

      if (kwc == NULL) {
        kwc = instance->GetClass();
        cvNativeAttributesContinuousMedianValues.SetSize(
            kwc->GetLoadedAttributeNumber()); // vecteur des valeurs finales de
                                              // medianes
        cvNativeAttributesContinuousMedianValues.Initialize();

        // initialisation du vecteur de vecteurs qui contiendra les valeurs
        // continues des attributs natifs
        continuousValues = new ObjectArray;
        continuousValues->SetSize(kwc->GetLoadedAttributeNumber());
        for (int i = 0; i < continuousValues->GetSize(); i++) {
          ContinuousVector *continuousValuesByAttribute = new ContinuousVector;
          continuousValuesByAttribute->SetSize(GetCount());
          continuousValuesByAttribute->Initialize();
          continuousValues->SetAt(i, continuousValuesByAttribute);
        }
      }

      // parcourir tous les attributs natifs de cette instance de cluster, et
      // stocker les valeurs continues dans les vecteurs correspondants

      for (int i = 0; i < nbNativeAttributes; i++) {

        const KWLoadIndex loadIndex = nativeAttributesLoadIndexes.GetAt(i);
        if (not(loadIndex.IsValid()))
          continue;

        KWAttribute *native =
            instance->GetClass()->GetAttributeAtLoadIndex(loadIndex);
        assert(native != NULL);

        if (native->GetType() == KWType::Continuous and
            instance->GetContinuousValueAt(native->GetLoadIndex()) !=
                KWContinuous::GetMissingValue()) {
          ContinuousVector *continuousValuesByAttribute =
              cast(ContinuousVector *, continuousValues->GetAt(i));
          assert(continuousValuesByAttribute != NULL);
          continuousValuesByAttribute->SetAt(
              idxInstance, instance->GetContinuousValueAt(loadIndex));
          cvNativeAttributesContinuousMedianValues.SetAt(
              i, cvNativeAttributesContinuousMedianValues.GetAt(i) +
                     instance->GetContinuousValueAt(loadIndex));
        }
      }
    }
    idxInstance++;
  }

  assert(kwc != NULL);

  // trier tous les vecteurs obtenus (un par attribut), par ordre croissant, et
  // en deduire la mediane
  for (int idxAttribute = 0; idxAttribute < continuousValues->GetSize();
       idxAttribute++) {

    ContinuousVector *continuousValuesByAttribute =
        cast(ContinuousVector *, continuousValues->GetAt(idxAttribute));

    if (continuousValuesByAttribute->GetSize() == 1) {
      // cas particulier : une seule instance dans la base
      cvNativeAttributesContinuousMedianValues.SetAt(
          idxAttribute, continuousValuesByAttribute->GetAt(0));
    } else {
      // si au moins 2 valeurs : calcul de la valeur mediane, selon la parite de
      // la taille du tableau de valeurs
      continuousValuesByAttribute->Sort();
      if (continuousValuesByAttribute->GetSize() % 2 == 0)
        cvNativeAttributesContinuousMedianValues.SetAt(
            idxAttribute, (continuousValuesByAttribute->GetAt(
                               continuousValuesByAttribute->GetSize() / 2 - 1) +
                           continuousValuesByAttribute->GetAt(
                               continuousValuesByAttribute->GetSize() / 2)) /
                              2);
      else
        cvNativeAttributesContinuousMedianValues.SetAt(
            idxAttribute, continuousValuesByAttribute->GetAt(
                              continuousValuesByAttribute->GetSize() / 2));
    }
  }

  if (continuousValues != NULL) {
    continuousValues->DeleteAll();
    delete continuousValues;
  }
}

Continuous KMCluster::GetNativeAttributeContinuousMeanValue(
    const KWAttribute *attr) const {

  if (cvNativeAttributesContinuousMeanValues.GetSize() == 0)
    return 0; // cas du cluster vide

  const ObjectDictionary &attributesIndexes =
      parameters->GetLoadedAttributesNames();

  const char *key = attr->GetName();

  if (attributesIndexes.Lookup(key) == NULL) {
    AddWarning("Can't get mean value for attribute " + attr->GetName() +
               ", because it's not loaded");
    return 0;
  }

  IntObject *io = cast(IntObject *, attributesIndexes.Lookup(key));
  assert(io->GetInt() < cvNativeAttributesContinuousMeanValues.GetSize());

  return cvNativeAttributesContinuousMeanValues.GetAt(io->GetInt());
}

Continuous KMCluster::GetNativeAttributeContinuousMedianValue(
    const KWAttribute *attr) const {

  if (cvNativeAttributesContinuousMedianValues.GetSize() == 0)
    return 0; // cas du cluster vide

  const ObjectDictionary &attributesIndexes =
      parameters->GetLoadedAttributesNames();

  const char *key = attr->GetName();

  if (attributesIndexes.Lookup(key) == NULL) {
    AddWarning("Can't get median value for attribute " + attr->GetName() +
               ", because it's not loaded");
    return 0;
  }

  IntObject *io = cast(IntObject *, attributesIndexes.Lookup(key));
  assert(io->GetInt() < cvNativeAttributesContinuousMedianValues.GetSize());

  return cvNativeAttributesContinuousMedianValues.GetAt(io->GetInt());
}

void KMCluster::IncrementInstancesWithMissingNativeValuesNumber(
    const KWObject *o) {

  if (ivMissingNativeValues.GetSize() == 0) {
    ivMissingNativeValues.SetSize(o->GetClass()->GetLoadedAttributeNumber());
    ivMissingNativeValues.Initialize();
  }

  const KWLoadIndexVector &nativeAttributesLoadIndexes =
      parameters->GetNativeAttributesLoadIndexes();
  const int size = nativeAttributesLoadIndexes.GetSize();

  for (int i = 0; i < size; i++) {

    const KWLoadIndex loadIndex = nativeAttributesLoadIndexes.GetAt(i);
    if (not(loadIndex.IsValid()))
      continue;

    KWAttribute *native = o->GetClass()->GetAttributeAtLoadIndex(loadIndex);
    assert(native != NULL);

    if (native->GetType() == KWType::Symbol and
        o->GetSymbolValueAt(loadIndex) == Symbol(""))
      ivMissingNativeValues.SetAt(i, ivMissingNativeValues.GetAt(i) + 1);
    else if (native->GetType() == KWType::Continuous and
             o->GetContinuousValueAt(loadIndex) ==
                 KWContinuous::GetMissingValue())
      ivMissingNativeValues.SetAt(i, ivMissingNativeValues.GetAt(i) + 1);
  }
}

int KMCluster::GetMissingValues(const KWAttribute *attr) const {

  if (ivMissingNativeValues.GetSize() == 0)
    return 0; // cas du cluster vide

  const ObjectDictionary &attributesIndexes =
      parameters->GetLoadedAttributesNames();

  const char *key = attr->GetName();

  if (attributesIndexes.Lookup(key) == NULL) {
    AddWarning("Can't get missing values number for attribute " +
               attr->GetName() + ", because it's not loaded");
    return 0;
  }

  IntObject *io = cast(IntObject *, attributesIndexes.Lookup(key));
  assert(io->GetInt() < ivMissingNativeValues.GetSize());

  return ivMissingNativeValues.GetAt(io->GetInt());
}

void KMCluster::SetStatisticsUpToDate(const bool b) { bStatisticsUpToDate = b; }

ALString KMCluster::GetLabel() const { return sLabel; }

void KMCluster::SetLabel(const ALString &s) { sLabel = s; }

void KMCluster::Write(ostream &ost) const {

  ost << endl
      << endl
      << "Cluster " << GetLabel() << ", address = " << this
      << ", index = " << iIndex << endl;
  ost << "Count = " << GetCount() << ", frequency = " << lFrequency << endl;
  ost << "Up to date stats : " << (bStatisticsUpToDate ? "yes" : "no") << endl;

  ost << "Non-zero MODELING centroid values, by attribute position : " << endl;

  for (int i = 0; i < cvModelingCentroidValues.GetSize(); i++)
    if (cvModelingCentroidValues.GetAt(i) != 0)
      ost << i << "\t" << cvModelingCentroidValues.GetAt(i) << endl;

  if (cvEvaluationCentroidValues.GetSize() > 0) {
    ost << endl
        << "Non-zero EVALUATION centroid values, by attribute position : "
        << endl;

    for (int i = 0; i < cvEvaluationCentroidValues.GetSize(); i++)
      if (cvEvaluationCentroidValues.GetAt(i) != 0)
        ost << i << "\t" << cvEvaluationCentroidValues.GetAt(i) << endl;
  }
  if (GetNearestCluster() != NULL)
    ost << "nearest cluster is " << GetNearestCluster() << endl;

  /*
  ost << endl << "Target probs :";
  GetTargetProbs().Write(ost);

  ost << endl
          << "Compactness : " << dCompactness << endl
          << "Majority target value " << sMajorityTargetValue << endl;

  ost << "Inerty intra L1 : " << cvInertyIntra.GetAt(KMParameters::L1Norm) <<
  endl; ost << "Inerty intra L2 : " << cvInertyIntra.GetAt(KMParameters::L2Norm)
  << endl; ost << "Inerty intra Cosine : " <<
  cvInertyIntra.GetAt(KMParameters::CosineNorm) << endl; ost << "Inerty intra L2
  by attributes load indexes : " << endl;
  cvInertyIntraL2ByAttributes.Write(ost);

  ost << endl << endl << "Instances :" << endl;
  NUMERIC key;
  Object * oCurrent;
  POSITION position = GetStartPosition();

  while (position != NULL){

  GetNextAssoc(position, key, oCurrent);
  KWObject * instance = static_cast<KWObject *>(oCurrent);
  instance->Write(ost);
  }
  */

  ost << endl << endl;
}

//////////////////////////////////////////////////////////
// Classe PLShared_Cluster
// Serialisation de la classe KMCluster

PLShared_Cluster::PLShared_Cluster() {}

PLShared_Cluster::~PLShared_Cluster() {}

void PLShared_Cluster::SetCluster(KMCluster *c) {
  require(c != NULL);
  SetObject(c);
}

KMCluster *PLShared_Cluster::GetCluster() {
  return cast(KMCluster *, GetObject());
}

void PLShared_Cluster::SerializeObject(PLSerializer *serializer,
                                       const Object *object) const {
  KMCluster *cluster;
  PLShared_ContinuousVector sharedContinuousVector;

  require(serializer != NULL);
  require(serializer->IsOpenForWrite());
  require(object != NULL);

  cluster = cast(KMCluster *, object);
  sharedContinuousVector.SerializeObject(serializer, &(cluster->cvTargetProbs));
  serializer->PutLongint(cluster->lFrequency);
}

void PLShared_Cluster::DeserializeObject(PLSerializer *serializer,
                                         Object *object) const {
  KMCluster *cluster;
  PLShared_ContinuousVector sharedContinuousVector;

  require(serializer->IsOpenForRead());

  cluster = cast(KMCluster *, object);

  // Deserialization des attributs
  sharedContinuousVector.DeserializeObject(serializer,
                                           &(cluster->cvTargetProbs));
  cluster->lFrequency = serializer->GetLongint();
}

Object *PLShared_Cluster::Create() const { return new KMCluster(NULL); }
