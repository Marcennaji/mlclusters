// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWPredictor.h"
#include "KWDRPredictor.h"
#include "KWClassStats.h"

#include "KWSTDatabaseTextFile.h"

#include "KMTrainedPredictor.h"
#include "KMTrainedClassifier.h"
#include "KMParameters.h"
#include "KMClustering.h"
#include "KMClusteringMiniBatch.h"
#include "KMPredictorReport.h"
#include "KMPredictorEvaluation.h"
#include "KMClassifierEvaluation.h"
#include "KMDRClassifier.h"
#include "KMDRLocalModelChooser.h"

class KMPredictorEvaluation;
class KMTrainedClassifier;
class KMTrainedPredictor;

/////////////////////////////////////////////////////////////////////
// Classe KMPredictor
/// Predicteur KMeans : permet l'apprentissage, le stockage du meilleur resultat d'un clustering, et la generation d'un dictionnaire de modelisation
//
//

class KMPredictor : public KWPredictor
{
public:
	// Constructeur
	KMPredictor();
	~KMPredictor();

	// Copie et duplication
	virtual void CopyFrom(const KMPredictor* aSource);
	KMPredictor* Clone() const;

	boolean IsTargetTypeManaged(int nType) const;

	/** implementation metghode virtuelle ancetre */
	virtual KWPredictorEvaluation* Evaluate(KWDatabase* database);

	KWPredictor* Create() const;

	/** Nom du predicteur */
	const ALString GetName() const;

	/** Prefixe du predicteur, utilisable pour le nommage de la classe en deploiement */
	const ALString GetPrefix() const;

	/** classifieur appris (mode supervise) */
	KMTrainedClassifier* GetTrainedClassifier();

	/** predicteur appris (non supervise) */
	KMTrainedPredictor* GetTrainedPredictor();

	/** resultat kmean d'un apprentissage (clusters, stats...) */
	KMClustering* GetBestTrainedClustering() const;

	const ObjectArray& GetLocalModelsPredictors() const;

	/** evaluer la memoire  necessaire pour traiter n instances en base de donnees, a l'aide du dico pass� en parametre */
	static longint ComputeRequiredMemory(const longint instancesNumber, const KWClass* kwc);

	/** parametrage du clustering */
	KMParameters* GetKMParameters() const;

	/** ajout d'un attribut de prediction dans un dico */
	static void AddPredictionAttributeToClass(KWTrainedPredictor* trainedPredictor, KWAttribute* attribute, KWClass* kwClass, const ALString label);

	/** nombre d'attributs d'un clustering */
	int GetClusteringVariablesNumber() const;

	static const char* ID_CLUSTER_METADATA;
	static const char* PREPARED_ATTRIBUTE_METADATA;
	static const char* CELL_INDEX_METADATA;
	static const char* ID_CLUSTER_LABEL;
	static const char* DISTANCE_CLUSTER_LABEL;
	static const char* CLUSTER_LABEL;
	static const char* PREDICTOR_NAME;
	static const char* GLOBAL_GRAVITY_CENTER_LABEL;

	///////////////////////////////////////////////////////
	//// Implementation
protected:

	/** Redefinition de la methode d'apprentissage */
	virtual boolean InternalTrain();

	boolean InternalTrain(KWDataPreparationClass* dataPreparationClass,
		ObjectArray* oaDataPreparationUsedAttributes);

	/** generation d'un modele supervise */
	boolean GenerateSupervisedModelingDictionary(KWTrainedClassifier* trainedKMean,
		KWDataPreparationClass*,
		ObjectArray* oaDataPreparationUsedAttributes,
		KWClass* localModelClass);

	/** generation d'un modele non supervise */
	boolean GenerateUnsupervisedModelingDictionary(KWDataPreparationClass* dataPreparationClass,
		ObjectArray* oaDataPreparationUsedAttributes);

	/** generation automatique d'un modele baseline, si aucun attribut informatif */
	boolean GenerateBaselineModelingDictionary(KWTrainedClassifier* trainedKMean,
		KWDataPreparationClass*,
		ObjectArray* oaDataPreparationUsedAttributes);

	/** Pour chaque attribut d'un modele, preciser quel est son centre de gravite global (i.e, la valeur de cet attribut dans le "cluster global") */
	void AddGlobalGravityCenters(KWClass* modeling);

	/** apprentissage : clustering kmean */
	bool ComputeAllReplicates(KWDataPreparationClass*);

	/** apprentissage : clustering mini-batch kmean */
	bool ComputeAllMiniBatchesReplicates(KWDataPreparationClass*);

	/** creation des attributs de distance, dans le dico de modelisation */
	boolean CreateDistanceClusterAttributes(KWDerivationRule* argminRule, KWClass* kwClass);

	/** creation des attributs de distance L1, dans le dico de modelisation */
	boolean CreateDistanceClusterAttributesL1(KWDerivationRule* argminRule, KWClass* kwClass);

	/** creation des attributs de distance L2, dans le dico de modelisation */
	boolean CreateDistanceClusterAttributesL2(KWDerivationRule* argminRule, KWClass* kwClass);

	/** creation des attributs de distance Cosinus, dans le dico de modelisation */
	boolean CreateDistanceClusterAttributesCosinus(KWDerivationRule* argminRule, KWClass* kwClass);

	/** creation des regles de derivation L1, dans le dico de modelisation */
	KWDerivationRule* GetL1NormDerivationRule(KWAttribute* attribute, const int idCluster);

	/** creation des regles de derivation L2, dans le dico de modelisation */
	KWDerivationRule* GetL2NormDerivationRule(KWAttribute* attribute, const int idCluster);

	/** creation du numerateur de la division (regle de derivation de la norme cosinus) */
	KWDerivationRule* GetCosineNormNumerator(KWClass* kwModelingClass, const int idCluster);

	/** creation du denominateur de la division (regle de derivation de la norme cosinus) */
	KWDerivationRule* GetCosineNormDenominator(KWClass* kwModelingClass, const int idCluster);

	/** creation des regles de derivation Cosinus, dans le dico de modelisation (numerateur) */
	KWDerivationRule* GetCosineNormNumeratorDerivationRule(KWAttribute* attribute, const int idCluster);

	/** creation des regles de derivation Cosinus, dans le dico de modelisation (denominateur partie 1) */
	KWDerivationRule* GetCosineNormDenominator1DerivationRule(KWAttribute* attribute, const int idCluster);

	/** creation des regles de derivation Cosinus, dans le dico de modelisation (denominateur partie 2) */
	KWDerivationRule* GetCosineNormDenominator2DerivationRule(KWAttribute* attribute);

	/** creation d'un attribut classifieur kmeans pour le dico de modelisation */
	KWAttribute* CreateGlobalModelClassifierAttribute(KWTrainedClassifier* trainedClassifier, KWAttribute* idClusterAttribute);

	/** creation d'un attribut classifieur baseline pour le dico de modelisation */
	KWAttribute* CreateBaselineModelClassifierAttribute(KWTrainedClassifier* trainedClassifier, ObjectArray* oaDataPreparationUsedAttributes);

	/** ajout des attributs de prediction du classifieur, dans le dico de modelisation */
	void AddClassifierPredictionAttributes(KWTrainedClassifier* trainedClassifier,
		KWAttribute* classifierAttribute);

	/** extraire les intervalles/modalites des attributs necessaires au calcul du level de clustering, a partir d'un dico */
	void ExtractPartitions(KWClass* aClass);

	/** repertorier les intervalles utilises pour generer les levels de clustering, dans le ModelingReport */
	void ExtractSourceConditionalInfoContinuous(const KWAttribute* attribute, const KWAttribute* nativeAttribute, KWClass* kwc);

	/** repertorier les modalites utilisees pour generer les levels de clustering, dans le ModelingReport */
	void ExtractSourceConditionalInfoCategorical(const KWAttribute* attribute, const KWAttribute* nativeAttribute, KWClass* kwc);

	/** Redefinition de la methode de creation du rapport */
	void CreatePredictorReport();

	/** creation d'un predicteur ou classifieur, selon si mode supervise ou non */
	void CreateTrainedPredictor();

	/** evaluer si la memoire est suffisante pour l'apprentissage */
	boolean HasSufficientMemoryForTraining(KWDataPreparationClass* dataPreparationClass, const int nInstancesNumber);

	/** customisation de la binarisation d'attributs */
	void AddPreparedBinarizationAttributes(ObjectArray* oaAddedAttributes, KWDataPreparationAttribute*);

	/** ConditionalInfo avec priors */
	void AddConditionalInfoWithPriorsAttributes(ObjectArray* oaAddedAttributes, KWDataPreparationAttribute*);

	/** Entropy */
	void AddEntropyAttributes(ObjectArray* oaAddedAttributes, KWDataPreparationAttribute*);

	/** EntropyWithPriors */
	void AddEntropyWithPriorsAttributes(ObjectArray* oaAddedAttributes, KWDataPreparationAttribute*);

	/** combiner une binarisation avec un ConditionalInfo */
	void AddHammingConditionalInfoAttributes(ObjectArray* oaAddedAttributes, KWDataPreparationAttribute*);

	KWAttribute* AddDataPreparationRuleAttribute(KWDerivationRule*, const ALString& sAttributePrefix, KWDataPreparationAttribute*);

	/** creation et gestion des attributs pretraites */
	bool GenerateRecodingDictionary(KWDataPreparationClass* dataPreparationClass, ObjectArray* oaDataPreparationFilteredAttributes);

	/** ne garder en memoire que les attributs utiles a l'apprentissage KMean */
	boolean PrepareLearningClass(KWClass* kwc, KWAttribute* targetAttribute);

	void AddAttributesMetaData(KWAttribute* nativeAttribute, KWAttribute* targetAttribute,
		ObjectArray& oaAddedAttributes);

	/** ajout des attributs necessaires a la production des rapports de frequences de modalit�s, et des levels de clustering */
	void AddCellIndexAttribute(KWClass* modelingClass, KWAttribute* preparedAttribute, const KWAttribute* nativeAttribute);

	/** apprentissage des modeles locaux */
	KWClass* TrainLocalModels(KWClass* recodingDictionary);

	/** creation d'un modele local a partir d'un cluster */
	KWPredictor* CreateLocalModelPredictorFromCluster(KMCluster*, KWClass* localModelClass, KWDatabase* localModelDatabase);

	/** creation de la database d'un modele local a partir d'un cluster */
	KWSTDatabaseTextFile* CreateLocalModelDatabaseFromCluster(KMCluster*, const KWClass* localModelClass);

	/** creation de l'attribut classifieur d'un modele local */
	KWAttribute* CreateLocalModelClassifierAttribute(KWClass* targetDictionary, KWClass* localModelClass, KWAttribute* idClusterAttribute);

	/** creation du dico d'un modele local */
	KWClass* CreateLocalModelClass(KWClass* recodingDictionary);

	/** modifier le dico d'un modele local, avant fusion dans le modele final */
	void PrepareLocalModelClassForMerging(KWClass* trainedLocalModelClass, ALString attributesPrefix);

	////////////////////   attributs  ////////////////////

	/** gestion des modeles locaux - liste de pointeurs sur KWClassStats */
	ObjectArray oaLocalModelsClassStats;

	/** gestion des modeles locaux - liste de pointeurs sur KWLearningSpec  */
	ObjectArray oaLocalModelsLearningSpecs;

	/** gestion des modeles locaux - liste de pointeurs sur KWSTDatabaseTextFile  */
	ObjectArray oaLocalModelsDatabases;

	/** gestion des modeles locaux - liste de pointeurs sur KWPredictor  */
	ObjectArray oaLocalModelsPredictors;

	/** gestion des modeles locaux - liste de pointeurs sur KWClass  */
	ObjectArray oaLocalModelsClasses;

	/** meilleur resultat de clustering observ� lors de l'apprentissage */
	KMClustering* kmBestTrainedClustering;

	/** Parametres du traitement kmean */
	KMParameters* parameters;

	/** nbre d'attributs utilises pour le clustering d'apprentissage */
	int iClusteringVariablesNumber;
};

inline const ObjectArray& KMPredictor::GetLocalModelsPredictors() const {
	return oaLocalModelsPredictors;
}

/** fonction de tri des attributs selon leur level */
int KMCompareLevel(const void* elem1, const void* elem2);


/** fonction de tri des attributs selon leur nom  */
int KMCompareAttributeName(const void* elem1, const void* elem2);
