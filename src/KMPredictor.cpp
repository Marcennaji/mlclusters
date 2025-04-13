// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMPredictor.h"
#include "KMParametersView.h"
#include "KMClusteringQuality.h"

#include <KWPredictorUnivariate.h>
#include "KWSTDatabaseTextFile.h"
#include "KMLearningProject.h"

#include <sstream>

KMPredictor::KMPredictor()
{
	parameters = new KMParameters();
	kmBestTrainedClustering = new KMClustering(parameters);
	iClusteringVariablesNumber = 0;
}

KMPredictor::~KMPredictor()
{
	// attention, ne pas detruire parameters avant kmBestTrainedClustering : ce dernier a besoin de ses parametres,
	// pour nettoyer ses donnees correctement
	delete kmBestTrainedClustering;
	delete parameters;

	oaLocalModelsClassStats.DeleteAll();
	oaLocalModelsLearningSpecs.DeleteAll();
	oaLocalModelsPredictors.DeleteAll();
	oaLocalModelsDatabases.DeleteAll();
	oaLocalModelsClasses.DeleteAll();
}

KMPredictor* KMPredictor::Clone() const
{
	KMPredictor* aClone;

	aClone = new KMPredictor;
	aClone->CopyFrom(this);
	return aClone;
}

void KMPredictor::CopyFrom(const KMPredictor* aSource)
{
	require(aSource != NULL);

	delete kmBestTrainedClustering;
	kmBestTrainedClustering = aSource->kmBestTrainedClustering->Clone();
	iClusteringVariablesNumber = aSource->iClusteringVariablesNumber;

	delete parameters;
	parameters = aSource->parameters->Clone();
}

boolean KMPredictor::IsTargetTypeManaged(int nType) const
{
	return (nType == KWType::None or nType == KWType::Symbol);
}

KWPredictor* KMPredictor::Create() const
{
	return new KMPredictor;
}

const ALString KMPredictor::GetPrefix() const
{
	return "KM";
}

const ALString KMPredictor::GetName() const
{
	return PREDICTOR_NAME;
}

KMParameters* KMPredictor::GetKMParameters() const
{
	return parameters;
}

int KMPredictor::GetClusteringVariablesNumber() const
{
	return iClusteringVariablesNumber;
}

KMClustering* KMPredictor::GetBestTrainedClustering() const {
	return kmBestTrainedClustering;
}

void KMPredictor::CreateTrainedPredictor()
{
	require(bIsTraining);
	require(trainedPredictor == NULL);

	// Creation du predicteur

	if (GetTargetAttributeType() == KWType::None)
		trainedPredictor = new KMTrainedPredictor;
	else
		trainedPredictor = new KMTrainedClassifier;

	check(trainedPredictor);

	// Memorisation de son nom
	trainedPredictor->SetName(GetName());
}


KMTrainedClassifier* KMPredictor::GetTrainedClassifier()
{
	require((IsTraining() and trainedPredictor != NULL) or IsTrained());
	require(GetTargetAttributeType() == KWType::Symbol);
	return cast(KMTrainedClassifier*, trainedPredictor);
}
KMTrainedPredictor* KMPredictor::GetTrainedPredictor()
{
	require((IsTraining() and trainedPredictor != NULL) or IsTrained());
	return cast(KMTrainedPredictor*, trainedPredictor);
}

// surcharge de la methode ancetre
boolean KMPredictor::InternalTrain()
{
	KWDataPreparationClass dataPreparationClass;
	// NB. : KWDataPreparationClass = Gestion des attributs et de leur recodage, en mode supervise ou non

	require(Check());
	require(GetClassStats() != NULL);
	require(GetClassStats()->IsStatsComputed());

	Global::SetSilentMode(false);

	if (parameters->GetVerboseMode())
		AddSimpleMessage("MLClusters internal version is " + ALString(INTERNAL_VERSION));

	// nettoyer les eventuels metadatas specifiques kmean qui pourraient �tre pr�sents dans le dico d'input
	GetClass()->RemoveAllAttributesMetaDataKey(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL);
	GetClass()->RemoveAllAttributesMetaDataKey(KMPredictor::DISTANCE_CLUSTER_LABEL);
	GetClass()->RemoveAllAttributesMetaDataKey(KMPredictor::CLUSTER_LABEL);
	GetClass()->RemoveAllAttributesMetaDataKey(KMParameters::KM_ATTRIBUTE_LABEL);
	GetClass()->RemoveAllAttributesMetaDataKey(KMPredictor::ID_CLUSTER_METADATA);
	GetClass()->RemoveAllAttributesMetaDataKey(KMPredictor::CELL_INDEX_METADATA);
	GetClass()->RemoveAllAttributesMetaDataKey(KMParametersView::DETAILED_STATISTICS_FIELD_NAME);
	GetClass()->RemoveAllAttributesMetaDataKey(KMParametersView::MAX_EVALUATED_ATTRIBUTES_NUMBER_FIELD_NAME);
	GetClass()->RemoveAllAttributesMetaDataKey(KMParametersView::CONTINUOUS_PREPROCESSING_FIELD_NAME);
	GetClass()->RemoveAllAttributesMetaDataKey(KMParametersView::CATEGORICAL_PREPROCESSING_FIELD_NAME);
	GetClass()->RemoveAllAttributesMetaDataKey(KMParametersView::CENTROID_TYPE_FIELD_NAME);
	GetClass()->RemoveAllAttributesMetaDataKey(KMParametersView::LOCAL_MODEL_TYPE_FIELD_NAME);
	GetClass()->RemoveAllAttributesMetaDataKey(PREPARED_ATTRIBUTE_METADATA);
	GetClass()->Compile();

	if (GetTargetAttributeType() == KWType::None)
		parameters->SetSupervisedMode(false);
	else {
		parameters->SetSupervisedMode(true);
		parameters->SetMainTargetModality(ALString(GetMainTargetModality()));
	}

	if (not parameters->Check())
		return false;

	// Parametrage de la preparation de donnees
	dataPreparationClass.SetLearningSpec(GetLearningSpec());

	// Generation de la classe de preparation des donnees
	dataPreparationClass.ComputeDataPreparationFromClassStats(GetClassStats());

	return InternalTrain(&dataPreparationClass, dataPreparationClass.GetDataPreparationAttributes());

}

boolean KMPredictor::InternalTrain(KWDataPreparationClass* dataPreparationClass,
	ObjectArray* oaDataPreparationUsedAttributes)
{
	ObjectArray oaFilteredDataPreparationAttributes;
	KWDataPreparationAttribute* dataPreparationAttribute;
	int nAttribute;
	boolean bOk = true;

	require(dataPreparationClass->CheckDataPreparation());
	require(GetPredictorReport() != NULL);

	// en mode benchmark, Train peut etre appel� plusieurs fois --> nettoyer les anciens resultats
	if (kmBestTrainedClustering->GetClusters()->GetSize() > 0) {
		delete kmBestTrainedClustering;
		kmBestTrainedClustering = new KMClustering(parameters);
		oaLocalModelsClassStats.DeleteAll();
		oaLocalModelsLearningSpecs.DeleteAll();
		oaLocalModelsPredictors.DeleteAll();
		oaLocalModelsDatabases.DeleteAll();
	}

	if (GetTargetAttributeType() == KWType::None) {
		// non supervis�
		require(GetTrainedPredictor() != NULL);
		require(GetTrainedPredictor()->GetPredictorClass() == NULL);
	}
	else {
		require(GetTrainedClassifier() != NULL);
		require(GetTrainedClassifier()->GetPredictorClass() == NULL);
	}

	GetTrainParameters()->SetMaxEvaluatedAttributeNumber(parameters->GetMaxEvaluatedAttributesNumber());

	if (not bOk) {
		dataPreparationClass->RemoveDataPreparation();
		return false;
	}

	// filtrage, tri et selection eventuelle des attributs

	for (nAttribute = 0; nAttribute < oaDataPreparationUsedAttributes->GetSize(); nAttribute++)
	{
		dataPreparationAttribute = cast(KWDataPreparationAttribute*, oaDataPreparationUsedAttributes->GetAt(nAttribute));

		if (not parameters->GetKeepNulLevelVariables() and dataPreparationAttribute->GetPreparedStats()->GetPreparedDataGridStats()->ComputeInformativeAttributeNumber() == 0) {
			continue;// nombre d'attributs informatifs de la datagrid de l'attribut prepare = 0 : on ignore cet attribut prepare dans tous les cas, sauf si on a specifie vouloir les garder via l'IHM
		}

		oaFilteredDataPreparationAttributes.Add(dataPreparationAttribute);
	}

	if (parameters->GetSupervisedMode()) {

		// Tri des attributs par importance predictive decroissante, et eventuellement, limitation au nombre max d'attributs
		oaFilteredDataPreparationAttributes.SetCompareFunction(KWDataPreparationAttributeCompareSortValue);
		oaFilteredDataPreparationAttributes.Sort();

		if (GetTrainParameters()->GetMaxEvaluatedAttributeNumber() > 0 and
			GetTrainParameters()->GetMaxEvaluatedAttributeNumber() < oaFilteredDataPreparationAttributes.GetSize())
			oaFilteredDataPreparationAttributes.SetSize(GetTrainParameters()->GetMaxEvaluatedAttributeNumber());

	}
	else {
		if (GetTrainParameters()->GetMaxEvaluatedAttributeNumber() > 0)
			AddWarning("Parameter 'Max number of evaluated variables' is applicable only in supervised mode. Parameter is ignored.");
	}

	// Initialisation de la classe du predicteur
	if (GetTargetAttributeType() == KWType::None)
		GetTrainedPredictor()->SetPredictorClass(dataPreparationClass->GetDataPreparationClass(), GetTargetAttributeType(), GetName());
	else
		GetTrainedClassifier()->SetPredictorClass(dataPreparationClass->GetDataPreparationClass(), GetTargetAttributeType(), GetName());

	// Memorisation du domaine initial, afin de permettre une lecture de la base avec le dico de modelisation
	KWClassDomain* initialCurrentDomain = KWClassDomain::GetCurrentDomain();

	// modifier le dico de modelisation pour y inclure les variables pretraitees
	// la lecture de  la base, lors de l'apprentissage, se fera a l'aide de ce dico
	if (not GenerateRecodingDictionary(dataPreparationClass, &oaFilteredDataPreparationAttributes)) {

		// aucun attribut informatif : en mode supervise, on genere tout de meme sur disque un modele baseline (classifieur majoritaire), mais on ne l'evaluera pas
		if (GetTargetAttributeType() != KWType::None)
			GenerateBaselineModelingDictionary(GetTrainedClassifier(), dataPreparationClass, oaDataPreparationUsedAttributes);

		dataPreparationClass->RemoveDataPreparation();
		// Restitution de l'etat initial
		KWClassDomain::SetCurrentDomain(initialCurrentDomain);
		KWClassDomain::GetCurrentDomain()->Compile();

		if (GetTargetAttributeType() == KWType::None)
			return false;// on arrete le traitement
		else
			return true;// on poursuit le traitement afin de generer quand meme le fichier disque du modele baseline classifieur majoritaire, meme s'il ne sera pas evalue
	}

	if (HasSufficientMemoryForTraining(dataPreparationClass, GetClassStats()->GetInstanceNumber()) and not parameters->GetMiniBatchMode())
		bOk = ComputeAllReplicates(dataPreparationClass);
	else {
		AddMessage("Using Kmean mini-batches mode.");
		parameters->SetMiniBatchMode(true);
		parameters->SetClustersCentersInitializationMethod(KMParameters::Random);
		if (GetTargetAttributeType() == KWType::Symbol and parameters->GetLocalModelType() != KMParameters::LocalModelType::None) {
			AddMessage("Due to mini-batches mode, no local models will be trained.");
			parameters->SetLocalModelType(KMParameters::LocalModelType::None);
		}
		bOk = ComputeAllMiniBatchesReplicates(dataPreparationClass);
	}

	KWClass* localModelClass = NULL;

	if (GetTargetAttributeType() == KWType::Symbol and
		parameters->GetLocalModelType() != KMParameters::LocalModelType::None) {

		localModelClass = TrainLocalModels(dataPreparationClass->GetDataPreparationClass());// apprendre les modeles "locaux" (propres a chaque cluster)

		if (localModelClass == NULL)
			bOk = false;
	}

	GetDatabase()->DeleteAll();

	// Restitution de l'etat initial
	KWClassDomain::SetCurrentDomain(initialCurrentDomain);
	KWClassDomain::GetCurrentDomain()->Compile();

	// generation du dictionnaire de modelisation
	if (bOk) {

		TaskProgression::DisplayLabel("Modeling dictionary generation...");

		if (parameters->GetVerboseMode()) {
			AddSimpleMessage("");
			AddSimpleMessage("Modeling dictionary generation");
		}

		cast(KMPredictorReport*, GetPredictorReport())->SetTrainedClustering(kmBestTrainedClustering);

		if (GetTargetAttributeType() == KWType::Symbol)
			bOk = GenerateSupervisedModelingDictionary(GetTrainedClassifier(), dataPreparationClass, &oaFilteredDataPreparationAttributes, localModelClass);
		else
			bOk = GenerateUnsupervisedModelingDictionary(dataPreparationClass, oaDataPreparationUsedAttributes);

		// ajout, dans le dico de modelisation, du centre de gravite global de chaque attribut KMean (necessaire a la production des rapports d'evaluation)
		if (bOk)
			AddGlobalGravityCenters(dataPreparationClass->GetDataPreparationClass());
	}

	dataPreparationClass->RemoveDataPreparation();

	GetPredictorReport()->SetUsedAttributeNumber(oaFilteredDataPreparationAttributes.GetSize());

	return bOk;
}

bool KMPredictor::ComputeAllReplicates(KWDataPreparationClass* dataPreparationClass) {

	Timer timer;
	timer.Start();

	bool bOk = false;

	if (not parameters->Check())
		return false;

	const KWAttribute* targetAttribute = dataPreparationClass->GetDataPreparationClass()->LookupAttribute(GetTargetAttributeName());

	int bestExecutionNumber = 1;

	GetDatabase()->ReadAll();

	TaskProgression::SetTitle("Clustering learning");

	if (parameters->GetVerboseMode()) {
		AddSimpleMessage(" ");
		AddSimpleMessage("Clustering parameters:");
		AddSimpleMessage("K = " + ALString(IntToString(parameters->GetKValue())));
		AddSimpleMessage("Min K value for post-optimisation training = " + ALString(IntToString(parameters->GetMinKValuePostOptimization())));
		AddSimpleMessage("Distance norm: " + ALString(parameters->GetDistanceTypeLabel()));
		AddSimpleMessage("Clusters initialization: " + ALString(parameters->GetClustersCentersInitializationMethodLabel()));
		AddSimpleMessage("Number of replicates: " + ALString(IntToString(parameters->GetLearningNumberOfReplicates())));
		AddSimpleMessage("Best replicate is based on " + ALString(parameters->GetReplicateChoiceLabel()));
		AddSimpleMessage("Max iterations number: " + ALString(IntToString(parameters->GetMaxIterations())));
		AddSimpleMessage("Max epsilon iterations number: " + ALString(IntToString(parameters->GetEpsilonMaxIterations())));
		AddSimpleMessage("Epsilon value: " + KMGetDisplayString(parameters->GetEpsilonValue()));
		AddSimpleMessage("Centroids type: " + ALString(parameters->GetCentroidTypeLabel()));
		AddSimpleMessage("Continuous preprocessing: " + ALString(parameters->GetContinuousPreprocessingTypeLabel(true)));
		AddSimpleMessage("Categorical preprocessing: " + ALString(parameters->GetCategoricalPreprocessingTypeLabel(true)));
		AddSimpleMessage("Preprocessing 'p' value (max intervals number): " + ALString(IntToString(parameters->GetPreprocessingMaxIntervalNumber())));
		AddSimpleMessage("Preprocessing 'q' value (max groups number): " + ALString(IntToString(parameters->GetPreprocessingMaxGroupNumber())));
	}

	ObjectArray* instances = GetDatabase()->GetObjects();

	const int nbInstances = instances->GetSize();
	if (parameters->GetKValue() > nbInstances) {
		AddWarning("K parameter (" + ALString(IntToString(parameters->GetKValue())) +
			") is greater than the number of instances in database (" + ALString(IntToString(nbInstances)) +
			"), setting K value to " + ALString(IntToString(nbInstances)));
		parameters->SetKValue(nbInstances);
	}

	const bool bSelectReplicatesOnEVA = (parameters->GetReplicateChoice() == KMParameters::EVA ? true : false);
	const bool bSelectReplicatesOnARIByClusters = (parameters->GetReplicateChoice() == KMParameters::ARIByClusters ? true : false);
	const bool bSelectReplicatesOnARIByClasses = (parameters->GetReplicateChoice() == KMParameters::ARIByClasses ? true : false);
	const bool bSelectReplicatesOnVariationOfInformation = (parameters->GetReplicateChoice() == KMParameters::VariationOfInformation ? true : false);
	const bool bSelectReplicatesOnLEVA = (parameters->GetReplicateChoice() == KMParameters::LEVA ? true : false);
	const bool bSelectReplicatesOnDaviesBouldin = (parameters->GetReplicateChoice() == KMParameters::DaviesBouldin ? true : false);
	const bool bSelectReplicatesOnPredictiveClustering = (parameters->GetReplicateChoice() == KMParameters::PredictiveClustering ? true : false);
	const bool bSelectReplicatesOnNormalizedMutualInformationByClusters = (parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClusters ? true : false);
	const bool bSelectReplicatesOnNormalizedMutualInformationByClasses = (parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClasses ? true : false);

	// on effectue plusieurs calculs kmean successifs (appel�s "replicates"), et on garde le meilleur resultat obtenu
	for (int iNumberOfReplicates = 0; iNumberOfReplicates < parameters->GetLearningNumberOfReplicates(); iNumberOfReplicates++) {

		KMClustering* currentClustering = new KMClustering(parameters);
		currentClustering->SetUsedSampleNumberPercentage(GetDatabase()->GetSampleNumberPercentage());

		// si ce n'est pas le premier replicate, recuperer les infos precedemment calculees, et dont ont est
		// sur qu'elles seront identiques lors des replicates suivants, afin de ne pas les recalculer inutilement
		if (iNumberOfReplicates > 0) {

			// recuperer les valeurs de modalit�s de la variable cible (mode supervis�)
			ObjectArray oaTargetAttributeValues;
			for (int i = 0; i < kmBestTrainedClustering->GetTargetAttributeValues().GetSize(); i++) {
				StringObject* value = new StringObject;
				value->SetString(cast(StringObject*, kmBestTrainedClustering->GetTargetAttributeValues().GetAt(i))->GetString());
				oaTargetAttributeValues.Add(value);
			}
			currentClustering->SetTargetAttributeValues(oaTargetAttributeValues);

			// recuperer les stats du cluster global
			assert(kmBestTrainedClustering->GetGlobalCluster() != NULL);
			currentClustering->SetGlobalCluster(kmBestTrainedClustering->GetGlobalCluster()->Clone());
		}

		if (parameters->GetLearningNumberOfReplicates() > 1 and parameters->GetVerboseMode()) {
			AddSimpleMessage(" ");
			AddSimpleMessage("*****************************************************");
			AddSimpleMessage("                     Replicate " + ALString(IntToString(iNumberOfReplicates + 1)));
			AddSimpleMessage("*****************************************************");
			AddSimpleMessage(" ");
		}

		ALString progressionLabel = "In progress : replicate " + ALString(IntToString(iNumberOfReplicates + 1));

		if (iNumberOfReplicates > 0) {
			progressionLabel += " (best execution is " + ALString(IntToString(bestExecutionNumber));

			if (bSelectReplicatesOnEVA)
				progressionLabel += ", with EVA = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetEVA())) + ")";
			else
				if (bSelectReplicatesOnARIByClusters)
					progressionLabel += ", with ARI by clusters = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetARIByClusters())) + ")";
				else
					if (bSelectReplicatesOnARIByClasses)
						progressionLabel += ", with ARI by classes = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetARIByClasses())) + ")";
					else
						if (bSelectReplicatesOnNormalizedMutualInformationByClusters)
							progressionLabel += ", with NMI by clusters = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters())) + ")";
						else
							if (bSelectReplicatesOnNormalizedMutualInformationByClasses)
								progressionLabel += ", with NMI by classes = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses())) + ")";
							else
								if (bSelectReplicatesOnVariationOfInformation)
									progressionLabel += ", with variation of information = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetVariationOfInformation())) + ")";
								else
									if (bSelectReplicatesOnLEVA)
										progressionLabel += ", with LEVA = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetLEVA())) + ")";
									else
										if (bSelectReplicatesOnDaviesBouldin)
											progressionLabel += ", with Davies-Bouldin = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetDaviesBouldin())) + ")";
										else
											if (bSelectReplicatesOnPredictiveClustering)
												progressionLabel += ", with Predictive Clustering value = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetPredictiveClustering())) + ")";
											else
												progressionLabel += ", with mean distance = " + ALString(DoubleToString(kmBestTrainedClustering->GetMeanDistance())) + ")";

		}

		TaskProgression::DisplayLabel(progressionLabel);

		int iOldSeed = GetRandomSeed();
		if (parameters->GetClustersCentersInitializationMethod() == KMParameters::Random and iNumberOfReplicates == 0)
			// si c'est le premier replicate et qu on utilise la methode d'initialisation random, on veut obtenir le meme tri des instances
			SetRandomSeed(1);

		// calcul kmean
		bOk = currentClustering->ComputeReplicate(instances, targetAttribute);

		// retablir l'ancienne valeur de seed si necessaire
		if (parameters->GetClustersCentersInitializationMethod() == KMParameters::Random and iNumberOfReplicates == 0)
			// si c'est le premier replicate et qu on utilise la methode d'initialisation random, on veut obtenir le meme tri des instances
			SetRandomSeed(iOldSeed);

		if (bOk) {

			if (iNumberOfReplicates == 0) {
				// si c'est le premier replicate, garder en memoire le resultat obtenu
				kmBestTrainedClustering->CopyFrom(currentClustering);

			}
			else {

				// si plusieurs replicates ont deja ete effectues, comparer cette execution avec la meilleure conservee auparavant

				bool isBestExecution = false;

				// selection du meilleur replicate sur le critere de l'EVA max
				if (bSelectReplicatesOnEVA and currentClustering->GetClusteringQuality()->GetEVA() > kmBestTrainedClustering->GetClusteringQuality()->GetEVA())
					isBestExecution = true;
				else
					// selection du meilleur replicate sur le critere de l'ARI max par cluster
					if (bSelectReplicatesOnARIByClusters and currentClustering->GetClusteringQuality()->GetARIByClusters() > kmBestTrainedClustering->GetClusteringQuality()->GetARIByClusters())
						isBestExecution = true;
					else
						// selection du meilleur replicate sur le critere de l'ARI max par classes
						if (bSelectReplicatesOnARIByClasses and currentClustering->GetClusteringQuality()->GetARIByClasses() > kmBestTrainedClustering->GetClusteringQuality()->GetARIByClasses())
							isBestExecution = true;
						else
							// selection du meilleur replicate sur le critere NMI clusters
							if (bSelectReplicatesOnNormalizedMutualInformationByClusters and currentClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters() > kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters())
								isBestExecution = true;
							else
								// selection du meilleur replicate sur le critere NMI classes
								if (bSelectReplicatesOnNormalizedMutualInformationByClasses and currentClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses() > kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses())
									isBestExecution = true;
								else
									// selection du meilleur replicate sur le critere de la variation min de l'information
									if (bSelectReplicatesOnVariationOfInformation and currentClustering->GetClusteringQuality()->GetVariationOfInformation() < kmBestTrainedClustering->GetClusteringQuality()->GetVariationOfInformation())
										isBestExecution = true;
									else
										// selection du meilleur replicate sur le critere du LEVA max
										if (bSelectReplicatesOnLEVA and currentClustering->GetClusteringQuality()->GetLEVA() > kmBestTrainedClustering->GetClusteringQuality()->GetLEVA())
											isBestExecution = true;
										else
											// selection du meilleur replicate sur le critere du Davis Bouldin min
											if (bSelectReplicatesOnDaviesBouldin and currentClustering->GetClusteringQuality()->GetDaviesBouldin() < kmBestTrainedClustering->GetClusteringQuality()->GetDaviesBouldin())
												isBestExecution = true;
											else
												// selection du meilleur replicate sur le critere PCC
												if (bSelectReplicatesOnPredictiveClustering and currentClustering->GetClusteringQuality()->GetPredictiveClustering() < kmBestTrainedClustering->GetClusteringQuality()->GetPredictiveClustering())
													isBestExecution = true;
												else
													// selection du meilleur replicate sur le critere de la distance min
													if (not bSelectReplicatesOnEVA and
														not bSelectReplicatesOnARIByClusters and
														not bSelectReplicatesOnARIByClasses and
														not bSelectReplicatesOnNormalizedMutualInformationByClusters and
														not bSelectReplicatesOnNormalizedMutualInformationByClasses and
														not bSelectReplicatesOnVariationOfInformation and
														not bSelectReplicatesOnLEVA and
														not bSelectReplicatesOnDaviesBouldin and
														not bSelectReplicatesOnPredictiveClustering) {

														if ((currentClustering->GetClustersDistanceSum(parameters->GetDistanceType()) < kmBestTrainedClustering->GetClustersDistanceSum(parameters->GetDistanceType())
															or kmBestTrainedClustering->GetClustersDistanceSum(parameters->GetDistanceType()) == 0.0)) {

															isBestExecution = true;

														}
													}

				if (isBestExecution) {

					if (parameters->GetVerboseMode())
						AddSimpleMessage("This is the best result so far.");

					bestExecutionNumber = iNumberOfReplicates + 1;
					kmBestTrainedClustering->CopyFrom(currentClustering);// ce resultat est le meilleur observ� jusqu'ici : le conserver

				}
			}
		}

		delete currentClustering;

		TaskProgression::DisplayProgression(((iNumberOfReplicates + 1) * 100) / parameters->GetLearningNumberOfReplicates());

		if (not bOk)
			break;
	}

	if (bOk and parameters->GetLearningNumberOfReplicates() > 1 and parameters->GetVerboseMode()) {

		AddSimpleMessage(" ");

		AddSimpleMessage("Best replicate is number " + ALString(IntToString(bestExecutionNumber)) + ":");
		AddSimpleMessage("\t- Mean distance is " + ALString(DoubleToString(kmBestTrainedClustering->GetMeanDistance())));
		AddSimpleMessage("\t- Davies-Bouldin index is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetDaviesBouldin())));

		if (targetAttribute != NULL) {
			AddSimpleMessage("\t- ARI by clusters is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetARIByClusters())));
			AddSimpleMessage("\t- Predictive clustering value is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetPredictiveClustering())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::EVA)
				AddSimpleMessage("\t- EVA is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetEVA())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::LEVA)
				AddSimpleMessage("\t- LEVA is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetLEVA())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::ARIByClasses)
				AddSimpleMessage("\t- ARI by classes is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetARIByClasses())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::VariationOfInformation)
				AddSimpleMessage("\t- Variation of information is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetVariationOfInformation())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClusters)
				AddSimpleMessage("\t- NMI by clusters is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClasses)
				AddSimpleMessage("\t- NMI by classes is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses())));
		}
		AddSimpleMessage(" ");
	}

	// si demand�, changer le centre de gravit� de chaque cluster, comme �tant l�instance la plus proche de son centre de gravit� virtuel
	if (bOk and parameters->GetCentroidType() == KMParameters::CentroidRealInstance) {

		AddSimpleMessage("Setting clusters's gravity centers to their center's nearest real instance");

		for (int idxCluster = 0; idxCluster < kmBestTrainedClustering->GetClusters()->GetSize(); idxCluster++) {

			KMCluster* cluster = cast(KMCluster*, kmBestTrainedClustering->GetClusters()->GetAt(idxCluster));
			assert(cluster != NULL);

			const KMClusterInstance* center = cluster->GetInstanceNearestToCentroid();

			if (center != NULL) { // NB. tenir compte du cas ou le cluster serait devenu vide
				cluster->InitializeModelingCentroidValues(center);
			}
		}
	}

	if (bOk) {

		kmBestTrainedClustering->AddInstancesToClusters(instances); // reaffecter les instances, car les sauvegardes du meilleur replicate ne gardent que les centroides des clusters obtenus et les stats, et pas les instances elles-memes

		// on ne touche pas aux stats qui avaient ete calculees lors de la sauvegarde du meilleur clustering : on indique ici que les stats des clusters ne doivent pas etre recalculees, malgre la reaffectation d'instances
		for (int i = 0; i < kmBestTrainedClustering->GetClusters()->GetSize(); i++) {
			KMCluster* c = cast(KMCluster*, kmBestTrainedClustering->GetClusters()->GetAt(i));
			c->SetStatisticsUpToDate(true);
		}

		if (parameters->GetSupervisedMode() and parameters->GetReplicatePostOptimization() == KMParameters::FastOptimization) {
			// supprimer certains centres de clusters, si cela a pour effet d'ameliorer l'EVA du clustering
			bOk = kmBestTrainedClustering->PostOptimize(instances, targetAttribute);
		}
	}

	// en supervis� et auto/auto, repertorier les modalites et intervalles des attributs (utilis�s pour generer les levels de clustering, dans le ModelingReport)
	if (bOk and
		GetTargetAttributeName() != "" and
		parameters->GetCategoricalPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed and
		parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed) {

		ExtractPartitions(dataPreparationClass->GetDataPreparationClass());

		// calculer les levels de clustering
		kmBestTrainedClustering->ComputeClusteringLevels(dataPreparationClass->GetDataPreparationClass(), GetClassStats()->GetAttributeStats(), kmBestTrainedClustering->GetClusters());
	}

	if (bOk and not (parameters->GetSupervisedMode() and parameters->GetReplicatePostOptimization())) {
		// le theoreme de Huygens est-il verifie ?
		if (parameters->GetDistanceType() == KMParameters::L2Norm and
			parameters->GetMaxIterations() == 0 and
			parameters->GetCentroidType() == KMParameters::CentroidVirtual and
			kmBestTrainedClustering->GetGlobalCluster() != NULL and
			not kmBestTrainedClustering->GetClusteringQuality()->CheckHuygensTheoremCorrectness(kmBestTrainedClustering->GetGlobalCluster()))
			AddWarning("Huygens theorem is not verified for this clustering.");
	}

	timer.Stop();

	if (parameters->GetVerboseMode()) {
		AddSimpleMessage(" ");
		AddSimpleMessage("Replicates total computing time : " + ALString(SecondsToString(timer.GetElapsedTime())));
		AddSimpleMessage(" ");
	}

	return bOk;
}


bool KMPredictor::ComputeAllMiniBatchesReplicates(KWDataPreparationClass* dataPreparationClass) {

	// modification de la valeur de l'echantillonnage de la base, pour qu'elle corresponde au nombre d'instances d'1 minibatch
	const int originalSamplePercentage = GetDatabase()->GetSampleNumberPercentage();// sauvegarder valeur originale
	int minibatchSamplePercentage = ((double)parameters->GetMiniBatchSize() / (double)GetDatabase()->GetSampleEstimatedObjectNumber()) * 100;

	if (minibatchSamplePercentage >= originalSamplePercentage) {
		AddWarning("Mini-batch size of " + ALString(IntToString(parameters->GetMiniBatchSize())) + " is too high, please try to decrease it.");
		return false;
	}

	if (not HasSufficientMemoryForTraining(dataPreparationClass, parameters->GetMiniBatchSize())) {
		AddWarning("Not enough memory to use a mini-batch size of " + ALString(IntToString(parameters->GetMiniBatchSize())) + ", please try to decrease it.");
		return false;
	}

	if (minibatchSamplePercentage == 0)
		minibatchSamplePercentage = 1;

	if (parameters->GetVerboseMode()) {
		AddMessage("Downsizing database sample to " + ALString(IntToString(minibatchSamplePercentage)) + "%, for mini-batches computing.");
	}

	int miniBatchesNumber = (originalSamplePercentage / minibatchSamplePercentage) + 1;// arrondir a l'entier superieur, et en faire au moins 2

	Timer timer;
	timer.Start();

	bool bOk = false;
	const KWAttribute* targetAttribute = dataPreparationClass->GetDataPreparationClass()->LookupAttribute(GetTargetAttributeName());

	int bestExecutionNumber = 1;

	TaskProgression::SetTitle("Mini-batches clustering learning");

	if (parameters->GetVerboseMode()) {
		AddSimpleMessage(" ");
		AddSimpleMessage("Clustering parameters (MINI BATCH MODE):");
		AddSimpleMessage("K = " + ALString(IntToString(parameters->GetKValue())));
		AddSimpleMessage("Mini-batches size: " + ALString(IntToString(parameters->GetMiniBatchSize())));
		AddSimpleMessage("Mini-batches number: " + ALString(IntToString(miniBatchesNumber)));
		AddSimpleMessage("Distance norm: " + ALString(parameters->GetDistanceTypeLabel()));
		AddSimpleMessage("Clusters initialization: " + ALString(parameters->GetClustersCentersInitializationMethodLabel()));
		AddSimpleMessage("Number of replicates: " + ALString(IntToString(parameters->GetLearningNumberOfReplicates())));
		AddSimpleMessage("Best replicate is based on " + ALString(parameters->GetReplicateChoiceLabel()));
		AddSimpleMessage("Max iterations number: " + ALString(IntToString(parameters->GetMaxIterations())));
		AddSimpleMessage("Max epsilon iterations number: " + ALString(IntToString(parameters->GetEpsilonMaxIterations())));
		AddSimpleMessage("Epsilon value: " + KMGetDisplayString(parameters->GetEpsilonValue()));
		AddSimpleMessage("Centroids type: " + ALString(parameters->GetCentroidTypeLabel()));
		AddSimpleMessage("Continuous preprocessing: " + ALString(parameters->GetContinuousPreprocessingTypeLabel(true)));
		AddSimpleMessage("Categorical preprocessing: " + ALString(parameters->GetCategoricalPreprocessingTypeLabel(true)));
		AddSimpleMessage("Preprocessing 'p' value (max intervals number): " + ALString(IntToString(parameters->GetPreprocessingMaxIntervalNumber())));
		AddSimpleMessage("Preprocessing 'q' value (max groups number): " + ALString(IntToString(parameters->GetPreprocessingMaxGroupNumber())));
	}

	const bool bSelectReplicatesOnEVA = (parameters->GetReplicateChoice() == KMParameters::EVA ? true : false);
	const bool bSelectReplicatesOnARIByClusters = (parameters->GetReplicateChoice() == KMParameters::ARIByClusters ? true : false);
	const bool bSelectReplicatesOnARIByClasses = (parameters->GetReplicateChoice() == KMParameters::ARIByClasses ? true : false);
	const bool bSelectReplicatesOnVariationOfInformation = (parameters->GetReplicateChoice() == KMParameters::VariationOfInformation ? true : false);
	const bool bSelectReplicatesOnLEVA = (parameters->GetReplicateChoice() == KMParameters::LEVA ? true : false);
	const bool bSelectReplicatesOnDaviesBouldin = (parameters->GetReplicateChoice() == KMParameters::DaviesBouldin ? true : false);
	const bool bSelectReplicatesOnPredictiveClustering = (parameters->GetReplicateChoice() == KMParameters::PredictiveClustering ? true : false);
	const bool bSelectReplicatesOnNormalizedMutualInformationByClusters = (parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClusters ? true : false);
	const bool bSelectReplicatesOnNormalizedMutualInformationByClasses = (parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClasses ? true : false);

	// calcul des stats globales sur le sample initial de la base
	KMClusteringMiniBatch* currentClustering = new KMClusteringMiniBatch(parameters);
	currentClustering->ComputeGlobalClusterStatistics(GetDatabase(), targetAttribute);

	// on effectue plusieurs replicates (chacun d'entre eux executera n iterations de mini-batchs), et on garde le meilleur resultat obtenu
	for (int iNumberOfReplicates = 0; iNumberOfReplicates < parameters->GetLearningNumberOfReplicates(); iNumberOfReplicates++) {

		if (iNumberOfReplicates > 0) {

			assert(kmBestTrainedClustering != NULL);

			currentClustering = new KMClusteringMiniBatch(parameters);
			currentClustering->SetUsedSampleNumberPercentage(originalSamplePercentage);

			//  ne pas recalculer inutilement les valeurs de modalit�s de la variable cible (mode supervis�)
			ObjectArray oaTargetAttributeValues;
			for (int i = 0; i < kmBestTrainedClustering->GetTargetAttributeValues().GetSize(); i++) {
				StringObject* value = new StringObject;
				value->SetString(cast(StringObject*, kmBestTrainedClustering->GetTargetAttributeValues().GetAt(i))->GetString());
				oaTargetAttributeValues.Add(value);
			}
			currentClustering->SetTargetAttributeValues(oaTargetAttributeValues);

			// ne pas recalculer inutilement les stats du cluster global
			assert(kmBestTrainedClustering->GetGlobalCluster() != NULL);
			currentClustering->SetGlobalCluster(kmBestTrainedClustering->GetGlobalCluster()->Clone());
		}

		if (parameters->GetLearningNumberOfReplicates() > 1 and parameters->GetVerboseMode()) {
			AddSimpleMessage(" ");
			AddSimpleMessage("*****************************************************");
			AddSimpleMessage("                     Replicate " + ALString(IntToString(iNumberOfReplicates + 1)) + " (mini-batches mode)");
			AddSimpleMessage("*****************************************************");
			AddSimpleMessage(" ");
		}

		ALString progressionLabel = "In progress : replicate " + ALString(IntToString(iNumberOfReplicates + 1));

		if (iNumberOfReplicates > 0) {
			progressionLabel += " (best execution is " + ALString(IntToString(bestExecutionNumber));

			if (bSelectReplicatesOnEVA)
				progressionLabel += ", with EVA = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetEVA())) + ")";
			else
				if (bSelectReplicatesOnARIByClusters)
					progressionLabel += ", with ARI by clusters = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetARIByClusters())) + ")";
				else
					if (bSelectReplicatesOnARIByClasses)
						progressionLabel += ", with ARI by classes = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetARIByClasses())) + ")";
					else
						if (bSelectReplicatesOnNormalizedMutualInformationByClusters)
							progressionLabel += ", with NMI by clusters = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters())) + ")";
						else
							if (bSelectReplicatesOnNormalizedMutualInformationByClasses)
								progressionLabel += ", with NMI by classes = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses())) + ")";
							else
								if (bSelectReplicatesOnVariationOfInformation)
									progressionLabel += ", with variation of information = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetVariationOfInformation())) + ")";
								else
									if (bSelectReplicatesOnLEVA)
										progressionLabel += ", with LEVA = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetLEVA())) + ")";
									else
										if (bSelectReplicatesOnDaviesBouldin)
											progressionLabel += ", with Davies-Bouldin = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetDaviesBouldin())) + ")";
										else
											if (bSelectReplicatesOnPredictiveClustering)
												progressionLabel += ", with Predictive Clustering value = " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetPredictiveClustering())) + ")";
											else
												progressionLabel += ", with mean distance = " + ALString(DoubleToString(kmBestTrainedClustering->GetMeanDistance())) + ")";

		}

		TaskProgression::DisplayLabel(progressionLabel);

		// calcul kmean par mini-batch
		bOk = currentClustering->ComputeReplicate(GetDatabase(), targetAttribute, miniBatchesNumber, originalSamplePercentage, minibatchSamplePercentage);

		if (bOk) {

			if (iNumberOfReplicates == 0) {
				// si c'est le premier replicate, garder en memoire le resultat obtenu
				kmBestTrainedClustering->CopyFrom(currentClustering);

			}
			else {

				// si plusieurs replicates ont deja ete effectues, comparer cette execution avec la meilleure conservee auparavant

				bool isBestExecution = false;

				// selection du meilleur replicate sur le critere de l'EVA max
				if (bSelectReplicatesOnEVA and currentClustering->GetClusteringQuality()->GetEVA() > kmBestTrainedClustering->GetClusteringQuality()->GetEVA())
					isBestExecution = true;
				else
					// selection du meilleur replicate sur le critere de l'ARI max par cluster
					if (bSelectReplicatesOnARIByClusters and currentClustering->GetClusteringQuality()->GetARIByClusters() > kmBestTrainedClustering->GetClusteringQuality()->GetARIByClusters())
						isBestExecution = true;
					else
						// selection du meilleur replicate sur le critere de l'ARI max par classes
						if (bSelectReplicatesOnARIByClasses and currentClustering->GetClusteringQuality()->GetARIByClasses() > kmBestTrainedClustering->GetClusteringQuality()->GetARIByClasses())
							isBestExecution = true;
						else
							// selection du meilleur replicate sur le critere NMI clusters
							if (bSelectReplicatesOnNormalizedMutualInformationByClusters and currentClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters() > kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters())
								isBestExecution = true;
							else
								// selection du meilleur replicate sur le critere NMI classes
								if (bSelectReplicatesOnNormalizedMutualInformationByClasses and currentClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses() > kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses())
									isBestExecution = true;
								else
									// selection du meilleur replicate sur le critere de la variation min de l'information
									if (bSelectReplicatesOnVariationOfInformation and currentClustering->GetClusteringQuality()->GetVariationOfInformation() < kmBestTrainedClustering->GetClusteringQuality()->GetVariationOfInformation())
										isBestExecution = true;
									else
										// selection du meilleur replicate sur le critere du LEVA max
										if (bSelectReplicatesOnLEVA and currentClustering->GetClusteringQuality()->GetLEVA() > kmBestTrainedClustering->GetClusteringQuality()->GetLEVA())
											isBestExecution = true;
										else
											// selection du meilleur replicate sur le critere du Davis Bouldin min
											if (bSelectReplicatesOnDaviesBouldin and currentClustering->GetClusteringQuality()->GetDaviesBouldin() < kmBestTrainedClustering->GetClusteringQuality()->GetDaviesBouldin())
												isBestExecution = true;
											else
												// selection du meilleur replicate sur le critere PCC
												if (bSelectReplicatesOnPredictiveClustering and currentClustering->GetClusteringQuality()->GetPredictiveClustering() < kmBestTrainedClustering->GetClusteringQuality()->GetPredictiveClustering())
													isBestExecution = true;
												else
													// selection du meilleur replicate sur le critere de la distance min
													if (not bSelectReplicatesOnEVA and
														not bSelectReplicatesOnARIByClusters and
														not bSelectReplicatesOnARIByClasses and
														not bSelectReplicatesOnNormalizedMutualInformationByClusters and
														not bSelectReplicatesOnNormalizedMutualInformationByClasses and
														not bSelectReplicatesOnVariationOfInformation and
														not bSelectReplicatesOnLEVA and
														not bSelectReplicatesOnDaviesBouldin and
														not bSelectReplicatesOnPredictiveClustering) {

														if ((currentClustering->GetClustersDistanceSum(parameters->GetDistanceType()) < kmBestTrainedClustering->GetClustersDistanceSum(parameters->GetDistanceType())
															or kmBestTrainedClustering->GetClustersDistanceSum(parameters->GetDistanceType()) == 0.0)) {

															isBestExecution = true;

														}
													}

				if (isBestExecution) {

					if (parameters->GetVerboseMode())
						AddSimpleMessage("This is the best result so far.");

					bestExecutionNumber = iNumberOfReplicates + 1;
					kmBestTrainedClustering->CopyFrom(currentClustering);// ce resultat est le meilleur observ� jusqu'ici : le conserver

				}
			}
		}

		delete currentClustering;

		TaskProgression::DisplayProgression(((iNumberOfReplicates + 1) * 100) / parameters->GetLearningNumberOfReplicates());

		if (not bOk)
			break;
	}

	if (bOk and parameters->GetLearningNumberOfReplicates() > 1 and parameters->GetVerboseMode()) {

		AddSimpleMessage(" ");

		AddSimpleMessage("Best replicate is number " + ALString(IntToString(bestExecutionNumber)) + ":");
		AddSimpleMessage("\t- Mean distance is " + ALString(DoubleToString(kmBestTrainedClustering->GetMeanDistance())));
		AddSimpleMessage("\t- Davies-Bouldin index is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetDaviesBouldin())));

		if (targetAttribute != NULL) {
			AddSimpleMessage("\t- ARI by clusters is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetARIByClusters())));
			AddSimpleMessage("\t- Predictive clustering value is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetPredictiveClustering())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::EVA)
				AddSimpleMessage("\t- EVA is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetEVA())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::LEVA)
				AddSimpleMessage("\t- LEVA is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetLEVA())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::ARIByClasses)
				AddSimpleMessage("\t- ARI by classes is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetARIByClasses())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::VariationOfInformation)
				AddSimpleMessage("\t- Variation of information is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetVariationOfInformation())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClusters)
				AddSimpleMessage("\t- NMI by clusters is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClasses)
				AddSimpleMessage("\t- NMI by classes is " + ALString(DoubleToString(kmBestTrainedClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses())));
		}
		AddSimpleMessage(" ");
	}

	// si demand�, changer le centre de gravit� de chaque cluster, comme �tant l�instance la plus proche de son centre de gravit� virtuel
	if (bOk and parameters->GetCentroidType() == KMParameters::CentroidRealInstance) {

		AddSimpleMessage("Setting clusters's gravity centers to their center's nearest real instance");

		for (int idxCluster = 0; idxCluster < kmBestTrainedClustering->GetClusters()->GetSize(); idxCluster++) {

			KMCluster* cluster = cast(KMCluster*, kmBestTrainedClustering->GetClusters()->GetAt(idxCluster));
			assert(cluster != NULL);

			KMClusterInstance* center = (KMClusterInstance*)cluster->GetInstanceNearestToCentroid();

			if (center != NULL) { // NB. tenir compte du cas ou le cluster serait devenu vide
				cluster->InitializeModelingCentroidValues(center);
			}
		}
	}

	// en supervis� et auto/auto, repertorier les modalites et intervalles des attributs (utilis�s pour generer les levels de clustering, dans le ModelingReport)
	if (bOk and
		GetTargetAttributeName() != "" and
		parameters->GetCategoricalPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed and
		parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed) {

		ExtractPartitions(dataPreparationClass->GetDataPreparationClass());

		// calculer les levels de clustering sur la BDD du minibatch
		kmBestTrainedClustering->ComputeClusteringLevels(GetDatabase(), dataPreparationClass->GetDataPreparationClass(), GetClassStats()->GetAttributeStats(), kmBestTrainedClustering->GetClusters());
	}

	timer.Stop();

	if (parameters->GetVerboseMode()) {
		AddSimpleMessage(" ");
		AddSimpleMessage("Replicates total computing time : " + ALString(SecondsToString(timer.GetElapsedTime())));
		AddSimpleMessage(" ");
	}

	return bOk;
}

KWClass* KMPredictor::TrainLocalModels(KWClass* recodingDictionary) {

	KWClass* localModelClass = CreateLocalModelClass(recodingDictionary); // creation du modele local et insertion dans le domaine courant

	if (localModelClass == NULL)
		return NULL;

	for (int idxCluster = 0; idxCluster < kmBestTrainedClustering->GetClusters()->GetSize(); idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, kmBestTrainedClustering->GetClusters()->GetAt(idxCluster));
		assert(cluster != NULL);

		AddSimpleMessage("");
		AddSimpleMessage(parameters->GetLocalModelTypeLabel() + " training on cluster " + ALString(IntToString(idxCluster + 1)));

		// creation de la base du modele local, regroupant les instances du cluster en cours
		KWSTDatabaseTextFile* localModelDatabase = CreateLocalModelDatabaseFromCluster(cluster, localModelClass);

		if (localModelDatabase != NULL) {

			// creation du modele local, et ComputeStats() sur sa base
			KWPredictor* localPredictor = CreateLocalModelPredictorFromCluster(cluster, localModelClass, localModelDatabase);
			assert(localPredictor != NULL);

			// apprentissage du modele local
			localPredictor->Train();

			localPredictor->GetPredictorReport()->SetLearningSpec(GetLearningSpec());// necessaire pour produire les modeling reports des modeles locaux

			// supprimer le fichier disque de la base du snb, devenu inutile
			ALString filename = localModelDatabase->GetDatabaseName();
			remove(filename);

			oaLocalModelsPredictors.Add(localPredictor);
			oaLocalModelsDatabases.Add(localModelDatabase);
		}

	}

	// retablir l'etat d'origine
	KWClassDomain::GetCurrentDomain()->RemoveClass(localModelClass->GetName());
	KWClassDomain::GetCurrentDomain()->InsertClass(recodingDictionary);

	return localModelClass;
}

KWClass* KMPredictor::CreateLocalModelClass(KWClass* recodingDictionary) {

	// ne retenir que les attributs natifs, ET qui ont ete retenus par la selection d'attributs en debut d'apprentissage

	KWClass* localModelClass = GetClass()->Clone();
	oaLocalModelsClasses.Add(localModelClass);

	StringVector* svUnwantedNativeAttributes = new StringVector;

	KWAttribute* attribute = localModelClass->GetHeadAttribute();

	while (attribute != NULL)
	{
		KWAttribute* recodingAttribute = recodingDictionary->LookupAttribute(attribute->GetName());
		assert(recodingAttribute != NULL);

		if (not recodingAttribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL))
			svUnwantedNativeAttributes->Add(attribute->GetName());

		localModelClass->GetNextAttribute(attribute);
	}

	for (int i = 0; i < svUnwantedNativeAttributes->GetSize(); i++) {
		localModelClass->DeleteAttribute(svUnwantedNativeAttributes->GetAt(i));
	}


	// verifier qu'il ne manque pas d'attribut (cas d'un attribut derive selectionne, faisant reference a un attribut non selectionne)
	boolean bAttributesOk = true;

	attribute = localModelClass->GetHeadAttribute();
	while (attribute != NULL)
	{
		// Test de l'attribut
		if (not attribute->Check()) {
			AddError("Local model : attribute " + attribute->GetName() + " is invalid (refers maybe to a non-existing attribute ?)");
			bAttributesOk = false;
		}

		// Attribut suivant
		localModelClass->GetNextAttribute(attribute);
	}


	delete svUnwantedNativeAttributes;

	if (not bAttributesOk) {
		// NB.: en cas d'attribut(s) invalide(s), le fait d'inserer la classe dans le domaine provoquera par la suite un plantage dans Learning. D'ou le controle en amont, sans utiliser
		// la methode KWClass::Check() , afin d'eviter l'insertion de la classe invalide dans le domaine

		// afficher le modele dans la fenetre de log :
		std::ostringstream oss;
		oss << "Local model is : " << endl;
		localModelClass->Write(oss);
		AddSimpleMessage(oss.str().c_str());
		return NULL;
	}

	// remplacer la classe du predicteur par la classe des modeles locaux, le temps de l'apprentissage de ces modeles
	KWClassDomain::GetCurrentDomain()->RemoveClass(recodingDictionary->GetName());
	KWClassDomain::GetCurrentDomain()->InsertClass(localModelClass);

	if (localModelClass->Check()) {
		localModelClass->Compile();
		return localModelClass;
	}
	else {
		// le dico local est invalide : retablir l'etat initial
		AddError("Can't create local model dictionary");
		KWClassDomain::GetCurrentDomain()->RemoveClass(localModelClass->GetName());
		KWClassDomain::GetCurrentDomain()->InsertClass(GetClass());
		delete localModelClass;
		return NULL;
	}
}


KWAttribute* KMPredictor::CreateLocalModelClassifierAttribute(KWClass* modelingClass, KWClass* localModelClass, KWAttribute* idClusterAttribute) {

	assert(trainedPredictor != NULL);
	assert(oaLocalModelsPredictors.GetSize() == kmBestTrainedClustering->GetClusters()->GetSize());

	const ALString sTargetProbMetaDataKey = "TargetProb";
	const ALString sTargetValuesMetaDataKey = "TargetValues";

	KWClassDomain* originalDomain = KWClassDomain::GetCurrentDomain();
	KWClassDomain* localModelsDomain = new KWClassDomain;
	localModelsDomain->SetName("localModelsDomain");
	KWClassDomain::SetCurrentDomain(localModelsDomain);
	KWClassDomain::GetCurrentDomain()->InsertClass(localModelClass);

	ObjectArray oaLocalModelClassifiersAttributes;

	// preparation des dictionnaires issus des apprentissages des modeles locaux
	for (int i = 0; i < oaLocalModelsPredictors.GetSize(); i++) {

		KWPredictor* predictor = cast(KWPredictor*, oaLocalModelsPredictors.GetAt(i));
		KWTrainedClassifier* trainedClassifier = predictor->GetTrainedClassifier();
		const ALString localModelAttributesPrefix = "localModel_" + ALString(IntToString(i)) + "_";
		PrepareLocalModelClassForMerging(trainedClassifier->GetPredictorClass(), localModelAttributesPrefix);// renommage des attributs pour eviter les doublons

		// insertion des attributs utiles, issus de l'apprentissage des modeles locaux, dans le dictionnaire cible
		KWAttribute* attribute = trainedClassifier->GetPredictorClass()->GetHeadAttribute();
		while (attribute != NULL)
		{
			if (modelingClass->LookupAttribute(attribute->GetName()) == NULL) {

				if (attribute->GetName() != trainedClassifier->GetPredictionAttribute()->GetName() and
					attribute->GetName() != trainedClassifier->GetScoreAttribute()->GetName()) {

					KWAttribute* insertedAttribute = attribute->Clone();

					insertedAttribute->GetMetaData()->SetStringValueAt(KMParametersView::LOCAL_MODEL_TYPE_FIELD_NAME, parameters->GetLocalModelTypeLabel());

					// Parcours des cles de l'attribut et renommage des metatags de prediction (pour eviter la presence de plusieurs attributs
					// possedant ces metatags, lors de la fusion des differents modeles locaux)
					for (int nKey = 0; nKey < insertedAttribute->GetMetaData()->GetKeyNumber(); nKey++)
					{
						ALString sKey = insertedAttribute->GetMetaData()->GetKeyAt(nKey);

						if (sKey.GetLength() > sTargetProbMetaDataKey.GetLength() and
							sKey.Left(sTargetProbMetaDataKey.GetLength()) == sTargetProbMetaDataKey)
						{
							ALString sIndex = sKey.Right(sKey.GetLength() - sTargetProbMetaDataKey.GetLength());
							int nIndex = StringToInt(sIndex);
							if (IntToString(nIndex) == sIndex)
								insertedAttribute->GetMetaData()->RemoveKey(sKey);
						}

						if (sKey == sTargetValuesMetaDataKey) {
							insertedAttribute->GetMetaData()->RemoveKey(sTargetValuesMetaDataKey);
							insertedAttribute->GetMetaData()->SetNoValueAt(sTargetValuesMetaDataKey + "_" + localModelAttributesPrefix);
						}
					}

					modelingClass->InsertAttribute(insertedAttribute);

				}

				if (attribute->GetStructureName() == "Classifier") {
					oaLocalModelClassifiersAttributes.Add(attribute);
				}
			}
			trainedClassifier->GetPredictorClass()->GetNextAttribute(attribute);
		}
	}

	KWClassDomain::GetCurrentDomain()->RemoveClass(localModelClass->GetName());

	// on aboutit � (exemple avec 2  clusters) :
	// Structure(Classifier)	LocalModelChooser(IdCluster, SNB1, SNB2,..., SymbolValueSet("Iris-versicolor", "Iris-virginica", "Iris-setosa"))	;
	// --> avec, en parametre, 1 classifieur par cluster. Le classifieur a utiliser pendant le deploiement sera determine par la valeur de l'id de cluster

	KWDerivationRuleOperand* operand;

	KMDRLocalModelChooser* modelChooserRule;

	modelChooserRule = new KMDRLocalModelChooser;
	modelChooserRule->DeleteAllOperands();

	// Ajout de l'operande id de cluster
	operand = new KWDerivationRuleOperand;
	operand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	operand->SetType(KWType::Continuous);
	operand->SetAttributeName(idClusterAttribute->GetName());
	modelChooserRule->AddOperand(operand);

	// Ajout d'un operande classifieur par cluster
	for (int i = 0; i < oaLocalModelClassifiersAttributes.GetSize(); i++)
	{
		KWAttribute* classifier = cast(KWAttribute*, oaLocalModelClassifiersAttributes.GetAt(i));
		operand = new KWDerivationRuleOperand;
		operand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		operand->SetType(KWType::Structure);
		operand->SetAttributeName(classifier->GetName());
		modelChooserRule->AddOperand(operand);
	}

	// Ajout d'un dernier operande pour les valeurs cible
	KWDRSymbolValueSet* symbolValueSetRule = new KWDRSymbolValueSet;
	symbolValueSetRule->DeleteAllOperands();

	// Parametrage des valeurs
	int nValueNumber = kmBestTrainedClustering->GetTargetAttributeValues().GetSize();
	assert(nValueNumber > 0);
	symbolValueSetRule->SetValueNumber(nValueNumber);

	for (int i = 0; i < nValueNumber; i++) {
		StringObject* value = cast(StringObject*, kmBestTrainedClustering->GetTargetAttributeValues().GetAt(i));
		Symbol s(value->GetString());
		symbolValueSetRule->SetValueAt(i, s);
	}
	operand = new KWDerivationRuleOperand;
	operand->SetDerivationRule(symbolValueSetRule);
	operand->SetOrigin(KWDerivationRuleOperand::OriginRule);
	operand->SetType(KWType::Structure);
	operand->SetStructureName(symbolValueSetRule->GetStructureName());

	modelChooserRule->AddOperand(operand);

	KWAttribute* modelChooserAttribute = new KWAttribute;
	modelChooserAttribute->SetName("localModel");
	modelChooserAttribute->SetDerivationRule(modelChooserRule);
	modelingClass->InsertAttribute(modelChooserAttribute);
	modelingClass->CompleteTypeInfo();

	modelingClass->Compile();

	KWClassDomain::SetCurrentDomain(originalDomain);

	delete localModelsDomain;

	return modelChooserAttribute;

}

void KMPredictor::PrepareLocalModelClassForMerging(KWClass* trainedLocalModelClass, ALString attributesPrefix) {

	assert(trainedLocalModelClass != NULL);

	KWAttribute* attribute = trainedLocalModelClass->GetHeadAttribute();

	while (attribute != NULL)
	{
		if (GetClass()->LookupAttribute(attribute->GetName()) != NULL or // ne pas renommer les attributs natifs, mais uniquement les attributs generes lors des apprentissages des modeles locaux
			attribute->GetConstMetaData()->IsKeyPresent("TargetVariable")) { // ignorer l'attribut cible
			trainedLocalModelClass->GetNextAttribute(attribute);
			continue;
		}

		trainedLocalModelClass->UnsafeRenameAttribute(attribute, attributesPrefix + attribute->GetName());

		trainedLocalModelClass->GetNextAttribute(attribute);
	}

	trainedLocalModelClass->Compile();
}

KWPredictor* KMPredictor::CreateLocalModelPredictorFromCluster(KMCluster* cluster, KWClass* localModelClass, KWDatabase* localModelDatabase) {

	KWPredictor* localModelPredictor = NULL;

	if (parameters->GetLocalModelType() == KMParameters::NB)
		localModelPredictor = new KWPredictorNaiveBayes;
	else {
		localModelPredictor = new SNBPredictorSelectiveNaiveBayes;
#ifdef _DEBUG
		((SNBPredictorSelectiveNaiveBayes*)localModelPredictor)->GetSelectionParameters()->SetOptimizationAlgorithm("FFW"); // pendant tests uniquement, pour gain de vitesse d'execution
		cout << "DEBUG FFW" << endl;
#endif
	}

	KWLearningSpec* spec = learningSpec->Clone(); // clonage de la spec, car chaque predicteur aura sa propre base

	if (parameters->GetLocalModelUseMODL()) {
		// forcer le pretraitement a MODL
		spec->GetPreprocessingSpec()->GetGrouperSpec()->SetSupervisedMethodName("MODL");
		spec->GetPreprocessingSpec()->GetDiscretizerSpec()->SetSupervisedMethodName("MODL");
	}

	spec->SetDatabase(localModelDatabase);
	spec->SetClass(localModelClass);
	localModelPredictor->SetLearningSpec(spec);
	oaLocalModelsLearningSpecs.Add(spec); // afin de gerer la desallocation en fin de traitement

	localModelPredictor->GetTrainParameters()->CopyFrom(GetTrainParameters());

	KWClassStats* kwcs = new KWClassStats;

	oaLocalModelsClassStats.Add(kwcs); // afin de gerer la desallocation en fin de traitement
	kwcs->SetLearningSpec(spec);

	KWAttributePairsSpec attributePairsSpec;
	attributePairsSpec.SetClassName(kwcs->GetClass()->GetName());
	kwcs->SetAttributePairsSpec(&attributePairsSpec);// requis depuis la v10, pour effectuer un KWClassStats::ComputeStats

	kwcs->ComputeStats();

	localModelPredictor->SetClassStats(kwcs);

	return localModelPredictor;

}

KWSTDatabaseTextFile* KMPredictor::CreateLocalModelDatabaseFromCluster(KMCluster* cluster, const KWClass* localModelClass) {

	// ecriture des individus du cluster, dans une database temporaire

	assert(cluster != NULL);
	assert(localModelClass != NULL);
	assert(localModelClass->IsCompiled());

	boolean bOk = FileService::CreateApplicationTmpDir();
	if (not bOk) {
		AddError("Can't create application temporary directory");
		return NULL;
	}

	KWSTDatabaseTextFile* dbTarget = new KWSTDatabaseTextFile;
	dbTarget->SetClassName(localModelClass->GetName());
	const ALString targetDatabaseFileName = FileService::CreateTmpFile("MLClusters_" + cluster->GetLabel(), this);
	dbTarget->SetDatabaseName(targetDatabaseFileName);

	if (not dbTarget->OpenForWrite()) {
		AddError("Can't create database '" + targetDatabaseFileName + "'");
		return NULL;
	}

	Global::ActivateErrorFlowControl();

	NUMERIC key;
	Object* oCurrent;
	POSITION position = cluster->GetStartPosition();

	int nbRecords = 0;

	while (position != NULL) {

		cluster->GetNextAssoc(position, key, oCurrent);
		KWObject* currentInstance = static_cast<KWObject *>(oCurrent);
		KWObject targetObject(localModelClass, nbRecords + 1);

		for (int idxAttribute = 0; idxAttribute < localModelClass->GetLoadedAttributeNumber(); idxAttribute++) {

			KWAttribute* attribute = localModelClass->GetLoadedAttributeAt(idxAttribute);

			Object* o = parameters->GetLoadedAttributesNames().Lookup(attribute->GetName());

			if (o == NULL) {
				// pendant tests uniquement
				AddWarning("attribute name " + attribute->GetName() + " not found in loaded attributes");
				continue;
			}
			IntObject* ioIndex = cast(IntObject*, o);
			KWLoadIndex loadIndex = parameters->GetLoadedAttributesLoadIndexes().GetAt(ioIndex->GetInt());

			if (attribute->GetType() == KWType::Continuous)
				targetObject.SetContinuousValueAt(attribute->GetLoadIndex(), currentInstance->GetContinuousValueAt(loadIndex));
			else
				if (attribute->GetType() == KWType::Symbol)
					targetObject.SetSymbolValueAt(attribute->GetLoadIndex(), currentInstance->GetSymbolValueAt(loadIndex));

		}

		dbTarget->Write(&targetObject);
		nbRecords++;
	}

	Global::DesactivateErrorFlowControl();

	if (not dbTarget->Close()) {
		AddError("Can't close database '" + targetDatabaseFileName + "'");
		return NULL;
	}

	return dbTarget;
}

void KMPredictor::ExtractPartitions(KWClass* kwc) {

	// Parcours du dictionnaire de modelisation pour identifier les attributs CellIndex necessaires
	KWAttribute* attribute = kwc->GetHeadAttribute();
	while (attribute != NULL)
	{
		if (attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::CELL_INDEX_METADATA)) {

			assert(attribute->GetDerivationRule() != NULL and attribute->GetDerivationRule()->GetName() == "CellIndex");

			// analyser la regle de derivation pour savoir a quel type de pretraitement on a affaire
			KWDerivationRule* kwdr = attribute->GetDerivationRule();
			KWDerivationRuleOperand* operand = kwdr->GetSecondOperand();

			KWAttribute* nativeAttribute = kwc->LookupAttribute(operand->GetAttributeName());

			assert(nativeAttribute != NULL);

			if (nativeAttribute->GetType() == KWType::Continuous)
				ExtractSourceConditionalInfoContinuous(attribute, nativeAttribute, kwc);
			else
				if (nativeAttribute->GetType() == KWType::Symbol)
					ExtractSourceConditionalInfoCategorical(attribute, nativeAttribute, kwc);

		}
		kwc->GetNextAttribute(attribute);
	}
}

boolean KMPredictor::GenerateUnsupervisedModelingDictionary(KWDataPreparationClass* dataPreparationClass,
	ObjectArray* oaDataPreparationUsedAttributes)
{
	require(Check());
	require(GetClassStats() != NULL);
	require(GetClassStats()->IsStatsComputed());
	require(GetTargetAttributeType() == KWType::None);
	require(IsTraining());
	boolean bOk = true;

	KWClass* kwModelingClass = dataPreparationClass->GetDataPreparationClass();

	KWDerivationRule* argminRule = new KWDRArgMin;
	argminRule->DeleteAllOperands();

	// creation des attributs DistanceCluster1 � DistanceClusterK dans le dico,
	// puis ajout de ces attributs, comme operandes a la regle argminRule
	bOk = CreateDistanceClusterAttributes(argminRule, kwModelingClass);

	if (not bOk)
		return false;

	argminRule->SetClassName(kwModelingClass->GetName());
	argminRule->CompleteTypeInfo(kwModelingClass);
	argminRule->Check();

	KWAttribute* idClusterAttribute = new KWAttribute;

	idClusterAttribute->SetName(kwModelingClass->BuildAttributeName(ID_CLUSTER_LABEL));
	idClusterAttribute->GetMetaData()->SetNoValueAt(KMPredictor::ID_CLUSTER_METADATA);
	if (parameters->GetWriteDetailedStatistics())
		idClusterAttribute->GetMetaData()->SetNoValueAt(KMParametersView::DETAILED_STATISTICS_FIELD_NAME);

	if (parameters->GetVerboseMode())
		idClusterAttribute->GetMetaData()->SetNoValueAt(KMParametersView::VERBOSE_MODE_FIELD_NAME);

	idClusterAttribute->GetMetaData()->SetStringValueAt(KMParametersView::CONTINUOUS_PREPROCESSING_FIELD_NAME, parameters->GetContinuousPreprocessingTypeLabel(false));
	idClusterAttribute->GetMetaData()->SetStringValueAt(KMParametersView::CATEGORICAL_PREPROCESSING_FIELD_NAME, parameters->GetCategoricalPreprocessingTypeLabel(false));

	idClusterAttribute->SetDerivationRule(argminRule);

	AddPredictionAttributeToClass(trainedPredictor, idClusterAttribute, kwModelingClass, ID_CLUSTER_METADATA);

	return bOk;

}
boolean KMPredictor::CreateDistanceClusterAttributes(KWDerivationRule* argminRule, KWClass* kwModelingClass)
{
	require(argminRule != NULL);
	require(kwModelingClass != NULL);
	require(kwModelingClass->IsIndexed());

	boolean bOk = true;

	TaskProgression::DisplayLabel("Modeling dictionary generation : distance cluster attributes");

	if (parameters->GetVerboseMode())
		AddSimpleMessage("Distance cluster attributes generation");

	if (parameters->GetDistanceType() == KMParameters::L1Norm)
		bOk = CreateDistanceClusterAttributesL1(argminRule, kwModelingClass);
	else
		if (parameters->GetDistanceType() == KMParameters::L2Norm)
			bOk = CreateDistanceClusterAttributesL2(argminRule, kwModelingClass);
		else
			if (parameters->GetDistanceType() == KMParameters::CosineNorm)
				bOk = CreateDistanceClusterAttributesCosinus(argminRule, kwModelingClass);

	return bOk;
}

boolean KMPredictor::CreateDistanceClusterAttributesL1(KWDerivationRule* argminRule, KWClass* kwModelingClass)
{
	// regle de la forme :
	// Sum(	Abs(Diff(Info1Page, 0.3553613)),
	//		Abs(Diff(Info2Page, 0.2474993)),
	//		Abs(Diff(Info1Pworkclass, 0.8558611)),
	//		Abs(Diff(Info2Pworkclass, 0.6916485))...)	;

	require(argminRule != NULL);
	require(kwModelingClass != NULL);
	require(kwModelingClass->IsIndexed());
	assert(parameters->GetDistanceType() == KMParameters::L1Norm);

	// on cree K attributs, de DistanceCluster1 a DistanceClusterK

	for (int k = 0; k < kmBestTrainedClustering->GetClusters()->GetSize(); k++) {

		KWAttribute* distanceAttribute = new KWAttribute;

		KMCluster* cluster = cast(KMCluster*, kmBestTrainedClustering->GetClusters()->GetAt(k));
		distanceAttribute->GetMetaData()->SetStringValueAt(CLUSTER_LABEL, cluster->GetLabel());

		const ALString attrName = kwModelingClass->BuildAttributeName(DISTANCE_CLUSTER_LABEL + ALString("_") + cluster->GetLabel() + "_L1");

		distanceAttribute->SetName(attrName);
		distanceAttribute->GetMetaData()->SetStringValueAt(DISTANCE_CLUSTER_LABEL, "L1");

		KWDerivationRule* sumRule = new KWDRSum;
		sumRule->DeleteAllOperands();

		int nProgression = k * 100 / kmBestTrainedClustering->GetClusters()->GetSize();
		TaskProgression::DisplayProgression(nProgression);

		if (TaskProgression::IsInterruptionRequested())
			break;

		int current = 0;

		KWAttribute* attribute = kwModelingClass->GetHeadAttribute();

		while (attribute != NULL)
		{
			current++;

			if (parameters->IsKMAttributeName(attribute->GetName()))
			{
				KWDerivationRuleOperand* sumOperand = new KWDerivationRuleOperand;
				sumOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
				sumOperand->SetType(KWType::Continuous);

				KWDerivationRule* rule = GetL1NormDerivationRule(attribute, k);// renvoie Abs(Diff(attributNatif_X, centreCorrespondantDansCluster_N)

				sumOperand->SetDerivationRule(rule);
				sumRule->AddOperand(sumOperand);
			}
			// Attribut suivant
			kwModelingClass->GetNextAttribute(attribute);
		}

		sumRule->SetClassName(kwModelingClass->GetName());
		sumRule->CompleteTypeInfo(kwModelingClass);
		sumRule->Check();

		distanceAttribute->SetDerivationRule(sumRule);

		AddPredictionAttributeToClass(trainedPredictor, distanceAttribute, kwModelingClass, attrName);

		// ajout du nouvel attribut DistanceCluster, en tant qu'operande de la regle argMin
		KWDerivationRuleOperand* argMinOperand = new KWDerivationRuleOperand;
		argMinOperand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		argMinOperand->SetType(KWType::Continuous);
		argMinOperand->SetAttributeName(distanceAttribute->GetName());
		argminRule->AddOperand(argMinOperand);

		// evaluer si on aura assez de memoire pour generer tous les attributs DistanceCluster suivants
		if (k == 0) {
			if (distanceAttribute->GetUsedMemory() * kmBestTrainedClustering->GetClusters()->GetSize() > RMResourceManager::GetRemainingAvailableMemory()) {
				AddError("Not enough memory for model generation");
				return false;
			}
		}
	}

	return true;
}
KWDerivationRule* KMPredictor::GetL1NormDerivationRule(KWAttribute* attribute, const int idCluster) {

	// en distance de norme L1, on renvoie une regle de derivation de la forme :
	//							Abs(Substract(attributNatif_X, centreCorrespondantDansCluster_N))

	assert(attribute->GetLoadIndex().IsValid());
	assert(idCluster != -1);

	// creation regle de derivation Abs
	KWDerivationRule* absRule = new KWDRAbs;
	absRule->DeleteAllOperands();
	KWDerivationRuleOperand* absOperand = new KWDerivationRuleOperand;
	absOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
	absOperand->SetType(KWType::Continuous);

	// creation regle de derivation Substract
	KWDRDiff* substractRule = new KWDRDiff;
	substractRule->DeleteAllOperands();

	KWDerivationRuleOperand* substractOperand1 = new KWDerivationRuleOperand;
	substractOperand1->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	substractOperand1->SetAttributeName(attribute->GetName());
	substractOperand1->SetType(KWType::Continuous);

	KWDerivationRuleOperand* substractOperand2 = new KWDerivationRuleOperand;
	substractOperand2->SetOrigin(KWDerivationRuleOperand::OriginConstant);
	substractOperand2->SetType(KWType::Continuous);
	const int rank = parameters->GetAttributeRankFromLoadIndex(attribute->GetLoadIndex());
	assert(rank != -1);
	substractOperand2->SetContinuousConstant(kmBestTrainedClustering->GetCluster(idCluster)->GetModelingCentroidValues().GetAt(rank));

	// ajouts des operandes cr��s et imbrication des regles de derivations entre elles
	substractRule->AddOperand(substractOperand1);
	substractRule->AddOperand(substractOperand2);

	absOperand->SetDerivationRule(substractRule);
	absRule->AddOperand(absOperand);

	return absRule;

}

boolean KMPredictor::CreateDistanceClusterAttributesL2(KWDerivationRule* argminRule, KWClass* kwModelingClass)
{
	require(argminRule != NULL);
	require(kwModelingClass != NULL);
	require(kwModelingClass->IsIndexed());
	assert(parameters->GetDistanceType() == KMParameters::L2Norm);

	// on cree K attributs, de DistanceCluster1 a DistanceClusterK

	for (int k = 0; k < kmBestTrainedClustering->GetClusters()->GetSize(); k++) {

		KWAttribute* distanceAttribute = new KWAttribute;

		KMCluster* cluster = cast(KMCluster*, kmBestTrainedClustering->GetClusters()->GetAt(k));
		distanceAttribute->GetMetaData()->SetStringValueAt(CLUSTER_LABEL, cluster->GetLabel());

		const ALString attrName = kwModelingClass->BuildAttributeName(DISTANCE_CLUSTER_LABEL + ALString("_") + cluster->GetLabel() + "_L2");

		distanceAttribute->SetName(attrName);
		distanceAttribute->GetMetaData()->SetStringValueAt(DISTANCE_CLUSTER_LABEL, "L2");

		KWDerivationRule* sumRule = new KWDRSum;
		sumRule->DeleteAllOperands();

		int nProgression = k * 100 / kmBestTrainedClustering->GetClusters()->GetSize();
		TaskProgression::DisplayProgression(nProgression);

		if (TaskProgression::IsInterruptionRequested())
			break;

		int current = 0;

		KWAttribute* attribute = kwModelingClass->GetHeadAttribute();

		while (attribute != NULL)
		{
			current++;

			if (parameters->IsKMAttributeName(attribute->GetName()))
			{
				KWDerivationRuleOperand* sumOperand = new KWDerivationRuleOperand;
				sumOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
				sumOperand->SetType(KWType::Continuous);

				KWDerivationRule* rule = GetL2NormDerivationRule(attribute, k);// renvoie Product(Substract(attributNatif_X, centreCorrespondantDansCluster_N),
																				//				   Substract(attributNatif_X, centreCorrespondantDansCluster_N

				sumOperand->SetDerivationRule(rule);
				sumRule->AddOperand(sumOperand);
			}
			// Attribut suivant
			kwModelingClass->GetNextAttribute(attribute);
		}

		sumRule->SetClassName(kwModelingClass->GetName());
		sumRule->CompleteTypeInfo(kwModelingClass);
		sumRule->Check();

		distanceAttribute->SetDerivationRule(sumRule);
		AddPredictionAttributeToClass(trainedPredictor, distanceAttribute, kwModelingClass, attrName);

		// ajout du nouvel attribut DistanceCluster, en tant qu'operande de la regle argMin
		KWDerivationRuleOperand* argMinOperand = new KWDerivationRuleOperand;
		argMinOperand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		argMinOperand->SetType(KWType::Continuous);
		argMinOperand->SetAttributeName(distanceAttribute->GetName());
		argminRule->AddOperand(argMinOperand);

		// evaluer si on aura assez de memoire pour generer tous les attributs DistanceCluster suivants
		if (k == 0) {
			if (distanceAttribute->GetUsedMemory() * kmBestTrainedClustering->GetClusters()->GetSize() > RMResourceManager::GetRemainingAvailableMemory()) {
				AddError("Not enough memory for model generation");
				return false;
			}
		}
	}
	return true;
}
KWDerivationRule* KMPredictor::GetL2NormDerivationRule(KWAttribute* attribute, const int idCluster) {

	// en distance de norme L2, on renvoie une regle de derivation de la forme :
	//							Product(Substract(attributNatif_X, centreCorrespondantDansCluster_N),
	//									Substract(attributNatif_X, centreCorrespondantDansCluster_N))

	assert(attribute->GetLoadIndex().IsValid());
	assert(idCluster != -1);

	// creation regle de derivation Product
	KWDerivationRule* productRule = new KWDRProduct;
	productRule->DeleteAllOperands();
	KWDerivationRuleOperand* productOperand1 = new KWDerivationRuleOperand;
	productOperand1->SetOrigin(KWDerivationRuleOperand::OriginRule);
	productOperand1->SetType(KWType::Continuous);

	// creation regle de derivation Substract
	KWDRDiff* substractRule = new KWDRDiff;
	substractRule->DeleteAllOperands();

	KWDerivationRuleOperand* substractOperand1 = new KWDerivationRuleOperand;
	substractOperand1->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	substractOperand1->SetAttributeName(attribute->GetName());
	substractOperand1->SetType(KWType::Continuous);

	KWDerivationRuleOperand* substractOperand2 = new KWDerivationRuleOperand;
	substractOperand2->SetOrigin(KWDerivationRuleOperand::OriginConstant);
	substractOperand2->SetType(KWType::Continuous);
	const int rank = parameters->GetAttributeRankFromLoadIndex(attribute->GetLoadIndex());
	assert(rank != -1);
	substractOperand2->SetContinuousConstant(kmBestTrainedClustering->GetCluster(idCluster)->GetModelingCentroidValues().GetAt(rank));

	// ajouts des operandes cr��s et imbrication des regles de derivations entre elles
	substractRule->AddOperand(substractOperand1);
	substractRule->AddOperand(substractOperand2);

	productOperand1->SetDerivationRule(substractRule);
	KWDerivationRuleOperand* productOperand2 = productOperand1->Clone();
	productRule->AddOperand(productOperand1);
	productRule->AddOperand(productOperand2);

	return productRule;

}
boolean KMPredictor::CreateDistanceClusterAttributesCosinus(KWDerivationRule* argminRule, KWClass* kwModelingClass)
{
	// cr�e des attributs DistanceCluster_CO avec des regles de la forme (exemple avec 2 attributs) :
	// Substract(1, Divide(	Sum(Product(Info1PSepalLength, 8.659907)), Product(Info2PSepalLength, 0.6488839)),
				//			Product(
				//					Power(Sum(Product(8.659907,8.659907),Product(0.6488839,0.6488839)),0.5),
				//					Power(Sum(Product(Info1PSepalLength,Info1PSepalLength),Product(Info2PSepalLength,Info2PSepalLength)),0.5)))

	require(argminRule != NULL);
	require(kwModelingClass != NULL);
	require(kwModelingClass->IsIndexed());
	assert(parameters->GetDistanceType() == KMParameters::CosineNorm);

	// on cree K attributs, de DistanceCluster1 a DistanceClusterK

	for (int k = 0; k < kmBestTrainedClustering->GetClusters()->GetSize(); k++) {

		int nProgression = k * 100 / kmBestTrainedClustering->GetClusters()->GetSize();
		TaskProgression::DisplayProgression(nProgression);

		if (TaskProgression::IsInterruptionRequested())
			break;

		KWAttribute* distanceAttribute = new KWAttribute;

		KMCluster* cluster = cast(KMCluster*, kmBestTrainedClustering->GetClusters()->GetAt(k));
		distanceAttribute->GetMetaData()->SetStringValueAt(CLUSTER_LABEL, cluster->GetLabel());

		const ALString attrName = kwModelingClass->BuildAttributeName(DISTANCE_CLUSTER_LABEL + ALString("_") + cluster->GetLabel() + "_CO");

		distanceAttribute->SetName(attrName);
		distanceAttribute->GetMetaData()->SetStringValueAt(DISTANCE_CLUSTER_LABEL, "CO");

		// calcul du numerateur de la division
		KWDerivationRule* sumRuleNumerator = GetCosineNormNumerator(kwModelingClass, k);

		// calcul du denominateur de la division
		KWDerivationRule* sumRuleDenominator = GetCosineNormDenominator(kwModelingClass, k);

		sumRuleNumerator->SetClassName(kwModelingClass->GetName());
		sumRuleNumerator->CompleteTypeInfo(kwModelingClass);
		sumRuleNumerator->Check();

		sumRuleDenominator->SetClassName(kwModelingClass->GetName());
		sumRuleDenominator->CompleteTypeInfo(kwModelingClass);
		sumRuleDenominator->Check();

		// construction de l'attribut de division
		KWDerivationRule* divideRule = new KWDRDivide;
		divideRule->DeleteAllOperands();

		KWDerivationRuleOperand* divideNumeratorOperand = new KWDerivationRuleOperand;
		divideNumeratorOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
		divideNumeratorOperand->SetType(KWType::Continuous);
		divideNumeratorOperand->SetDerivationRule(sumRuleNumerator);

		KWDerivationRuleOperand* divideDenominatorOperand = new KWDerivationRuleOperand;
		divideDenominatorOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
		divideDenominatorOperand->SetType(KWType::Continuous);
		divideDenominatorOperand->SetDerivationRule(sumRuleDenominator);

		divideRule->AddOperand(divideNumeratorOperand);
		divideRule->AddOperand(divideDenominatorOperand);

		// construction de l'attribut de soustraction
		KWDerivationRule* substractRule = new KWDRDiff;
		substractRule->DeleteAllOperands();

		KWDerivationRuleOperand* substractOperand1 = new KWDerivationRuleOperand;
		substractOperand1->SetOrigin(KWDerivationRuleOperand::OriginConstant);
		substractOperand1->SetType(KWType::Continuous);
		substractOperand1->SetContinuousConstant(1);

		KWDerivationRuleOperand* substractOperand2 = new KWDerivationRuleOperand;
		substractOperand2->SetOrigin(KWDerivationRuleOperand::OriginRule);
		substractOperand2->SetType(KWType::Continuous);
		substractOperand2->SetDerivationRule(divideRule);

		substractRule->AddOperand(substractOperand1);
		substractRule->AddOperand(substractOperand2);

		distanceAttribute->SetDerivationRule(substractRule);
		AddPredictionAttributeToClass(trainedPredictor, distanceAttribute, kwModelingClass, attrName);

		// ajout du nouvel attribut DistanceCluster, en tant qu'operande de la regle argMin
		KWDerivationRuleOperand* argMinOperand = new KWDerivationRuleOperand;
		argMinOperand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		argMinOperand->SetType(KWType::Continuous);
		argMinOperand->SetAttributeName(distanceAttribute->GetName());
		argminRule->AddOperand(argMinOperand);

		// evaluer si on aura assez de memoire pour generer tous les attributs DistanceCluster suivants
		if (k == 0) {
			if (distanceAttribute->GetUsedMemory() * kmBestTrainedClustering->GetClusters()->GetSize() > RMResourceManager::GetRemainingAvailableMemory()) {
				AddError("Not enough memory for model generation");
				return false;
			}
		}
	}

	return true;
}

KWDerivationRule* KMPredictor::GetCosineNormNumerator(KWClass* kwModelingClass, const int idCluster) {

	// renvoie une r�gle du style : Sum(Product(Info1PSepalLength, 8.659907)), Product(Info2PSepalLength, 0.6488839))

	KWDerivationRule* sumRuleNumerator = new KWDRSum;
	sumRuleNumerator->DeleteAllOperands();

	int current = 0;
	KWAttribute* attribute = kwModelingClass->GetHeadAttribute();

	while (attribute != NULL)
	{
		current++;

		if (TaskProgression::IsInterruptionRequested())
			break;

		if (parameters->IsKMAttributeName(attribute->GetName()))
		{
			KWDerivationRuleOperand* productOperand = new KWDerivationRuleOperand;
			productOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
			productOperand->SetType(KWType::Continuous);
			KWDerivationRule* rule = GetCosineNormNumeratorDerivationRule(attribute, idCluster);// renvoie Product(Info1PSepalLength, 8.659907)
			productOperand->SetDerivationRule(rule);
			sumRuleNumerator->AddOperand(productOperand);
		}
		// Attribut suivant
		kwModelingClass->GetNextAttribute(attribute);
	}

	return sumRuleNumerator;
}

KWDerivationRule* KMPredictor::GetCosineNormDenominator(KWClass* kwModelingClass, const int idCluster) {

	// Renvoie une r�gle du style :
	//			Product(
	//					Power(Sum(Product(8.659907,8.659907),Product(0.6488839,0.6488839)),0.5),
	//					Power(Sum(Product(Info1PSepalLength,Info1PSepalLength),Product(Info2PSepalLength,Info2PSepalLength)),0.5)))

	KWDerivationRule* sumRuleDenominator1 = new KWDRSum;
	sumRuleDenominator1->DeleteAllOperands();

	KWDerivationRule* sumRuleDenominator2 = new KWDRSum;
	sumRuleDenominator2->DeleteAllOperands();

	int current = 0;
	KWAttribute* attribute = kwModelingClass->GetHeadAttribute();

	while (attribute != NULL)
	{
		current++;

		if (TaskProgression::IsInterruptionRequested())
			break;

		if (parameters->IsKMAttributeName(attribute->GetName()))
		{
			KWDerivationRuleOperand* productOperand1 = new KWDerivationRuleOperand;
			productOperand1->SetOrigin(KWDerivationRuleOperand::OriginRule);
			productOperand1->SetType(KWType::Continuous);
			KWDerivationRule* rule1 = GetCosineNormDenominator1DerivationRule(attribute, idCluster);// renvoie par exemple Product(8.659907,8.659907)
			productOperand1->SetDerivationRule(rule1);
			sumRuleDenominator1->AddOperand(productOperand1);

			KWDerivationRuleOperand* productOperand2 = new KWDerivationRuleOperand;
			productOperand2->SetOrigin(KWDerivationRuleOperand::OriginRule);
			productOperand2->SetType(KWType::Continuous);
			KWDerivationRule* rule2 = GetCosineNormDenominator2DerivationRule(attribute);// renvoie par exemple Product(Info1PSepalLength,Info1PSepalLength)
			productOperand2->SetDerivationRule(rule2);
			sumRuleDenominator2->AddOperand(productOperand2);

		}
		// Attribut suivant
		kwModelingClass->GetNextAttribute(attribute);
	}

	// calcul de la premiere racine carree :
	KWDerivationRule* power1Rule = new KWDRPower;
	power1Rule->DeleteAllOperands();

	KWDerivationRuleOperand* power1Operand1 = new KWDerivationRuleOperand;
	power1Operand1->SetOrigin(KWDerivationRuleOperand::OriginRule);
	power1Operand1->SetType(KWType::Continuous);
	power1Operand1->SetDerivationRule(sumRuleDenominator1);

	KWDerivationRuleOperand* power1Operand2 = new KWDerivationRuleOperand;
	power1Operand2->SetOrigin(KWDerivationRuleOperand::OriginConstant);
	power1Operand2->SetType(KWType::Continuous);
	power1Operand2->SetContinuousConstant(0.5);

	power1Rule->AddOperand(power1Operand1);
	power1Rule->AddOperand(power1Operand2);

	// calcul de la deuxieme racine carree :
	KWDerivationRule* power2Rule = new KWDRPower;
	power2Rule->DeleteAllOperands();
	KWDerivationRuleOperand* power2Operand1 = new KWDerivationRuleOperand;
	power2Operand1->SetOrigin(KWDerivationRuleOperand::OriginRule);
	power2Operand1->SetType(KWType::Continuous);
	power2Operand1->SetDerivationRule(sumRuleDenominator2);

	KWDerivationRuleOperand* power2Operand2 = new KWDerivationRuleOperand;
	power2Operand2->SetOrigin(KWDerivationRuleOperand::OriginConstant);
	power2Operand2->SetType(KWType::Continuous);
	power2Operand2->SetContinuousConstant(0.5);

	power2Rule->AddOperand(power2Operand1);
	power2Rule->AddOperand(power2Operand2);

	// produit final
	KWDRProduct* productRuleDenominator = new KWDRProduct;
	productRuleDenominator->DeleteAllOperands();

	KWDerivationRuleOperand* productRuleOperand1 = new KWDerivationRuleOperand;
	productRuleOperand1->SetOrigin(KWDerivationRuleOperand::OriginRule);
	productRuleOperand1->SetType(KWType::Continuous);
	productRuleOperand1->SetDerivationRule(power1Rule);
	productRuleDenominator->AddOperand(productRuleOperand1);

	KWDerivationRuleOperand* productRuleOperand2 = new KWDerivationRuleOperand;
	productRuleOperand2->SetOrigin(KWDerivationRuleOperand::OriginRule);
	productRuleOperand2->SetType(KWType::Continuous);
	productRuleOperand2->SetDerivationRule(power2Rule);
	productRuleDenominator->AddOperand(productRuleOperand2);

	return productRuleDenominator;

}

KWDerivationRule* KMPredictor::GetCosineNormNumeratorDerivationRule(KWAttribute* attribute, const int idCluster) {

	// on renvoie une regle de derivation de la forme Product(Info1PSepalLength, 8.659907)

	assert(attribute != NULL);
	assert(attribute->GetLoadIndex().IsValid());
	assert(idCluster != -1);

	// creation regle de derivation Product
	KWDerivationRule* productRule = new KWDRProduct;
	productRule->DeleteAllOperands();
	KWDerivationRuleOperand* productOperand1 = new KWDerivationRuleOperand;
	productOperand1->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	productOperand1->SetAttributeName(attribute->GetName());
	productOperand1->SetType(KWType::Continuous);

	KWDerivationRuleOperand* productOperand2 = new KWDerivationRuleOperand;
	productOperand2->SetOrigin(KWDerivationRuleOperand::OriginConstant);
	productOperand2->SetType(KWType::Continuous);
	const int rank = parameters->GetAttributeRankFromLoadIndex(attribute->GetLoadIndex());
	assert(rank != -1);
	productOperand2->SetContinuousConstant(kmBestTrainedClustering->GetCluster(idCluster)->GetModelingCentroidValues().GetAt(rank));

	// ajouts des operandes cr��s et imbrication des regles de derivations entre elles
	productRule->AddOperand(productOperand1);
	productRule->AddOperand(productOperand2);

	return productRule;

}

KWDerivationRule* KMPredictor::GetCosineNormDenominator1DerivationRule(KWAttribute* attribute, const int idCluster) {

	// on renvoie une regle de derivation de la forme Product(8.659907,8.659907)

	assert(attribute != NULL);
	assert(attribute->GetLoadIndex().IsValid());
	assert(idCluster != -1);

	// creation regle de derivation Product
	KWDerivationRule* productRule = new KWDRProduct;
	productRule->DeleteAllOperands();

	KWDerivationRuleOperand* productOperand1 = new KWDerivationRuleOperand;
	productOperand1->SetOrigin(KWDerivationRuleOperand::OriginConstant);
	productOperand1->SetType(KWType::Continuous);
	const int rank = parameters->GetAttributeRankFromLoadIndex(attribute->GetLoadIndex());
	assert(rank != -1);
	productOperand1->SetContinuousConstant(kmBestTrainedClustering->GetCluster(idCluster)->GetModelingCentroidValues().GetAt(rank));

	KWDerivationRuleOperand* productOperand2 = productOperand1->Clone();

	productRule->AddOperand(productOperand1);
	productRule->AddOperand(productOperand2);

	return productRule;

}

KWDerivationRule* KMPredictor::GetCosineNormDenominator2DerivationRule(KWAttribute* attribute) {

	// renvoie une regle de derivation de la forme Product(Info1PSepalLength,Info1PSepalLength)

	assert(attribute != NULL);

	// creation regle de derivation Product
	KWDerivationRule* productRule = new KWDRProduct;
	productRule->DeleteAllOperands();

	KWDerivationRuleOperand* productOperand1 = new KWDerivationRuleOperand;
	productOperand1->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	productOperand1->SetAttributeName(attribute->GetName());
	productOperand1->SetType(KWType::Continuous);

	KWDerivationRuleOperand* productOperand2 = productOperand1->Clone();

	productRule->AddOperand(productOperand1);
	productRule->AddOperand(productOperand2);

	return productRule;

}
void KMPredictor::AddPredictionAttributeToClass(KWTrainedPredictor* trainedPredictor, KWAttribute* attribute, KWClass* kwClass, const ALString label)
{
	require(trainedPredictor != NULL);
	require(attribute != NULL);
	require(kwClass != NULL);
	require(label != "");

	// Completion des informations pour finaliser la specification
	attribute->CompleteTypeInfo(kwClass);

	attribute->SetName(kwClass->BuildAttributeName(attribute->GetName()));

	kwClass->InsertAttribute(attribute);

	kwClass->Compile();

	// Ajout dans les specifications d'apprentissage
	KWPredictionAttributeSpec* predictionAttributeSpec = new KWPredictionAttributeSpec;

	ALString sNewLabel;
	for (int i = 0; i < label.GetLength(); i++)
	{
		char c = label.GetAt(i);
		if (c == '-')
			sNewLabel += '_';
		else
			sNewLabel += c;
	}

	predictionAttributeSpec->SetLabel(sNewLabel);
	predictionAttributeSpec->SetType(attribute->GetType());
	predictionAttributeSpec->SetMandatory(true);
	predictionAttributeSpec->SetEvaluation(false);

	predictionAttributeSpec->SetAttribute(attribute);

	trainedPredictor->AddPredictionAttributeSpec(predictionAttributeSpec);

}

boolean KMPredictor::GenerateSupervisedModelingDictionary(KWTrainedClassifier* trainedKMean,
	KWDataPreparationClass* dataPreparationClass,
	ObjectArray* oaDataPreparationUsedAttributes,
	KWClass* localModelClass)
{
	KWAttribute* targetValuesAttribute = NULL;
	KWAttribute* classifier = NULL;
	boolean bOk = true;

	require(trainedKMean != NULL);
	require(trainedKMean->GetPredictorClass() != NULL);
	require(dataPreparationClass != NULL);
	require(oaDataPreparationUsedAttributes != NULL);
	require(GetTargetDescriptiveStats()->GetValueNumber() > 0);

	KWClass* kwModelingClass = dataPreparationClass->GetDataPreparationClass();

	KWDerivationRule* argminRule = new KWDRArgMin;
	argminRule->DeleteAllOperands();

	// creation des attributs DistanceCluster1 � DistanceClusterK dans le dico,
	// puis ajout de ces attributs, comme operandes a la regle argminRule
	bOk = CreateDistanceClusterAttributes(argminRule, kwModelingClass);

	if (not bOk)
		return false;

	argminRule->SetClassName(kwModelingClass->GetName());
	argminRule->CompleteTypeInfo(kwModelingClass);
	argminRule->Check();

	KWAttribute* idClusterAttribute = new KWAttribute;

	idClusterAttribute->SetName(kwModelingClass->BuildAttributeName(ID_CLUSTER_LABEL));
	idClusterAttribute->SetDerivationRule(argminRule);

	if (parameters->GetWriteDetailedStatistics())
		idClusterAttribute->GetMetaData()->SetNoValueAt(KMParametersView::DETAILED_STATISTICS_FIELD_NAME);

	if (parameters->GetVerboseMode())
		idClusterAttribute->GetMetaData()->SetNoValueAt(KMParametersView::VERBOSE_MODE_FIELD_NAME);

	idClusterAttribute->GetMetaData()->SetStringValueAt(KMParametersView::CONTINUOUS_PREPROCESSING_FIELD_NAME, parameters->GetContinuousPreprocessingTypeLabel(false));
	idClusterAttribute->GetMetaData()->SetStringValueAt(KMParametersView::CATEGORICAL_PREPROCESSING_FIELD_NAME, parameters->GetCategoricalPreprocessingTypeLabel(false));

	// Completion automatique des informations de la classe (nom de classe par regle...)
	trainedKMean->GetPredictorClass()->CompleteTypeInfo();

	AddPredictionAttributeToClass(trainedPredictor, idClusterAttribute, kwModelingClass, ID_CLUSTER_METADATA);

	//// Memorisation de la reference a l'attribut cible
	trainedKMean->SetTargetAttribute(
		trainedKMean->GetPredictorClass()->LookupAttribute(GetTargetAttributeName()));

	if (parameters->GetLocalModelType() != KMParameters::LocalModelType::None)
		// traitement des modeles issu des apprentissages locaux
		classifier = CreateLocalModelClassifierAttribute(kwModelingClass, localModelClass, idClusterAttribute);
	else
		// Ajout de l'attribut de prediction (modele global)
		classifier = CreateGlobalModelClassifierAttribute(trainedKMean, idClusterAttribute);

	// attribut memorisant les valeurs cibles
	targetValuesAttribute = dataPreparationClass->GetDataPreparationTargetAttribute()->GetPreparedAttribute();
	trainedKMean->SetTargetValuesAttribute(targetValuesAttribute);

	// Ajout des attributs de prediction
	AddClassifierPredictionAttributes(trainedKMean, classifier);

	TaskProgression::DisplayProgression(100);

	return true;
}

void KMPredictor::AddCellIndexAttribute(KWClass* modelingClass, KWAttribute* preparedAttribute, const KWAttribute* nativeAttribute) {

	// pour chaque attribut pr�par� de type datagrid, ajouter un attribut de type ValueIndex, necessaire pour produire
	// les rapports de frequences de modalit�s. Necessaire aussi pour produire les levels de clustering figurant
	// dans le modeling report.

	assert(preparedAttribute->GetStructureName() == "DataGrid" and not preparedAttribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL));

	const double level = preparedAttribute->GetConstMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey());

	assert(level > 0);

	// Creation d'une regle pour indexer les cellules
	KWDRCellIndex* valueIndexRule = new KWDRCellIndex;
	valueIndexRule->GetFirstOperand()->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	valueIndexRule->GetFirstOperand()->SetAttributeName(preparedAttribute->GetName());

	valueIndexRule->DeleteAllVariableOperands();
	KWDerivationRuleOperand* operand = new KWDerivationRuleOperand;
	valueIndexRule->AddOperand(operand);
	operand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	operand->SetType(nativeAttribute->GetType());
	operand->SetAttributeName(nativeAttribute->GetName());
	valueIndexRule->CompleteTypeInfo(modelingClass);

	// Ajout de l'attribut de calcul des index de valeurs cibles
	KWAttribute* valueIndexAttribute = new KWAttribute;
	valueIndexAttribute->SetName(modelingClass->BuildAttributeName("CellIndex" + preparedAttribute->GetName()));
	valueIndexAttribute->SetDerivationRule(valueIndexRule);
	valueIndexAttribute->SetType(valueIndexAttribute->GetDerivationRule()->GetType());
	valueIndexAttribute->GetMetaData()->SetNoValueAt(CELL_INDEX_METADATA);
	valueIndexAttribute->GetMetaData()->SetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey(), level);
	valueIndexAttribute->CompleteTypeInfo(modelingClass);
	modelingClass->InsertAttribute(valueIndexAttribute);

	preparedAttribute->GetMetaData()->SetStringValueAt(PREPARED_ATTRIBUTE_METADATA, nativeAttribute->GetName());// pouvoir retrouver facilement le nom d'attribut natif, lors d'une �valuation de mod�le

}

KWAttribute* KMPredictor::CreateGlobalModelClassifierAttribute(KWTrainedClassifier* trainedClassifier, KWAttribute* idClusterAttribute)
{

	// on aboutit � (exemple avec 2  clusters) :
	// Structure(Classifier) KMClass =	KMDRClassifier(IdCluster, ContinuousVector(0.3, 0.4, 0.3), ContinuousVector(0.5, 0.4, 0.1), SymbolValueSet("Iris-setosa", "Iris-versicolor", "Iris-virginica"));
	// --> avec 1 ContinuousVector par cluster, contenant les r�partitions des valeurs de la variable cible dans ce cluster

	KWDerivationRuleOperand* operand;

	KMDRClassifier* classifierRule;
	KWAttribute* classifierAttribute;

	require(trainedClassifier != NULL);

	classifierRule = new KMDRClassifier;
	classifierRule->DeleteAllOperands();

	// Ajout de l'operande id de cluster
	operand = new KWDerivationRuleOperand;
	operand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	operand->SetAttributeName(idClusterAttribute->GetName());
	classifierRule->AddOperand(operand);


	// Ajout d'un operande ContinuousVector par cluster
	for (int i = 0; i < kmBestTrainedClustering->GetClusters()->GetSize(); i++)
	{
		KMCluster* cluster = cast(KMCluster*, kmBestTrainedClustering->GetClusters()->GetAt(i));
		KWDRContinuousVector* continuousVectorRule = new KWDRContinuousVector;
		continuousVectorRule->SetValueNumber(cluster->GetTargetProbs().GetSize());
		for (int iTarget = 0; iTarget < cluster->GetTargetProbs().GetSize(); iTarget++)
			continuousVectorRule->SetValueAt(iTarget, cluster->GetTargetProbs().GetAt(iTarget));

		operand = new KWDerivationRuleOperand;
		operand->SetOrigin(KWDerivationRuleOperand::OriginRule);
		operand->SetDerivationRule(continuousVectorRule);
		operand->SetType(continuousVectorRule->GetType());
		classifierRule->AddOperand(operand);
	}

	// Ajout d'un dernier operande pour les valeurs cible
	KWDRSymbolValueSet* symbolValueSetRule = new KWDRSymbolValueSet;
	symbolValueSetRule->DeleteAllOperands();

	// Parametrage des valeurs
	int nValueNumber = kmBestTrainedClustering->GetTargetAttributeValues().GetSize();
	assert(nValueNumber > 0);
	symbolValueSetRule->SetValueNumber(nValueNumber);

	for (int i = 0; i < nValueNumber; i++) {
		StringObject* value = cast(StringObject*, kmBestTrainedClustering->GetTargetAttributeValues().GetAt(i));
		Symbol s(value->GetString());
		symbolValueSetRule->SetValueAt(i, s);
	}
	operand = new KWDerivationRuleOperand;
	operand->SetDerivationRule(symbolValueSetRule);
	operand->SetOrigin(KWDerivationRuleOperand::OriginRule);
	operand->SetType(KWType::Structure);
	operand->SetStructureName(symbolValueSetRule->GetStructureName());

	classifierRule->AddOperand(operand);

	// Creation d'un attribut de classification
	classifierAttribute = trainedClassifier->CreatePredictionAttribute(
		GetPrefix() + GetTargetAttributeName(), classifierRule);

	return classifierAttribute;
}


KWAttribute* KMPredictor::CreateBaselineModelClassifierAttribute(KWTrainedClassifier* trainedClassifier, ObjectArray* oaDataPreparationUsedAttributes)
{

	boolean bTrace = false;
	int nAttribute;
	KWDataPreparationAttribute* dataPreparationAttribute;
	KWDerivationRuleOperand* operand;
	KWDRDataGridStats* dgsRule;
	KWDRNBClassifier* classifierRule;
	KWAttribute* classifierAttribute;

	require(trainedClassifier != NULL);
	require(oaDataPreparationUsedAttributes != NULL);

	classifierRule = new KWDRNBClassifier;
	classifierRule->DeleteAllVariableOperands();

	// Ajout d'un attribut de type grille de donnees par attribut prepare
	for (nAttribute = 0; nAttribute < oaDataPreparationUsedAttributes->GetSize(); nAttribute++)
	{
		dataPreparationAttribute = cast(KWDataPreparationAttribute*,
			oaDataPreparationUsedAttributes->GetAt(nAttribute));

		// Affichage des statistiques de preparation
		if (bTrace)
		{
			cout << dataPreparationAttribute->GetObjectLabel() << endl;
			cout << *dataPreparationAttribute->GetPreparedStats()->GetPreparedDataGridStats() << endl;
		}

		// Creation d'une regle DataGridStats
		dgsRule = dataPreparationAttribute->CreatePreparedStatsRule();

		// Ajout d'un operande DataGridStats au predicteur
		operand = new KWDerivationRuleOperand;
		operand->SetOrigin(KWDerivationRuleOperand::OriginRule);
		operand->SetDerivationRule(dgsRule);
		operand->SetType(dgsRule->GetType());
		operand->SetStructureName(dgsRule->GetStructureName());
		classifierRule->AddOperand(operand);
	}

	// Ajout d'un dernier operande pour les valeurs cible
	operand = new KWDerivationRuleOperand;
	operand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	operand->SetAttributeName(trainedClassifier->GetTargetValuesAttribute()->GetName());
	operand->SetType(trainedClassifier->GetTargetValuesAttribute()->GetType());
	operand->SetStructureName(trainedClassifier->GetTargetValuesAttribute()->GetStructureName());
	classifierRule->AddOperand(operand);

	// Creation d'un attribut de classification
	classifierAttribute = trainedClassifier->CreatePredictionAttribute(
		GetPrefix() + GetTargetAttributeName(), classifierRule);
	return classifierAttribute;
}

void KMPredictor::AddClassifierPredictionAttributes(KWTrainedClassifier* trainedClassifier,
	KWAttribute* classifierAttribute)
{
	KWDRTargetValue* predictionRule;
	KWDRTargetProb* scoreRule;
	KWDRTargetProbAt* targetProbRule;
	KWAttribute* predictionAttribute;
	KWAttribute* scoreAttribute;
	KWAttribute* targetProbAttribute;
	int nTarget;


	require(trainedClassifier != NULL);
	require(classifierAttribute != NULL);
	require(trainedClassifier->GetPredictorClass()->LookupAttribute(classifierAttribute->GetName()) == classifierAttribute);
	require(trainedClassifier->GetTargetValuesAttribute() != NULL);
	require(trainedClassifier->GetPredictorClass()->LookupAttribute(
		trainedClassifier->GetTargetValuesAttribute()->GetName()) ==
		trainedClassifier->GetTargetValuesAttribute());

	// Creation d'une regle de prediction de la valeur cible
	// creation de la ligne :  Symbol	PredictedClass	 = TargetValue(KMClass)	;
	predictionRule = new KWDRTargetValue;
	predictionRule->GetFirstOperand()->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	predictionRule->GetFirstOperand()->SetAttributeName(classifierAttribute->GetName());

	// Ajout de l'attribut dans la classe
	predictionAttribute = trainedClassifier->CreatePredictionAttribute(
		"Predicted" + GetTargetAttributeName(), predictionRule);
	trainedClassifier->SetPredictionAttribute(predictionAttribute);

	// Creation d'une regle de prediction du score
	//  creation de la ligne :  Continuous	ScoreClass	 = TargetProb(KMClass)	;
	scoreRule = new KWDRTargetProb;
	scoreRule->GetFirstOperand()->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
	scoreRule->GetFirstOperand()->SetAttributeName(classifierAttribute->GetName());

	// Ajout de l'attribut dans la classe
	scoreAttribute = trainedClassifier->CreatePredictionAttribute(
		"Score" + GetTargetAttributeName(), scoreRule);
	trainedClassifier->SetScoreAttribute(scoreAttribute);

	// Memorisation du nombre d'attribut de prediction des probabilites cibles
	assert(GetTargetValueStats()->GetAttributeNumber() == 1);
	assert(GetTargetValueStats()->GetAttributeAt(0)->GetAttributeType() == KWType::Symbol);
	assert(GetTargetValueStats()->GetAttributeAt(0)->ArePartsSingletons());
	const KWDGSAttributeSymbolValues* targetValues = cast(const KWDGSAttributeSymbolValues*, GetTargetValueStats()->GetAttributeAt(0));
	trainedClassifier->SetTargetValueNumber(targetValues->GetValueNumber());

	// Creation d'attributs pour les probabilites conditionnelles par valeur cible
	for (nTarget = 0; nTarget < targetValues->GetValueNumber(); nTarget++)
	{
		// Creation d'une regle de prediction de la probabilite cible pour une valeur donnee
		// creation des lignes du type :   Numerical	Probclassless	 = TargetProbAt(KMclass, "less")	; <TargetProb1="less">
		targetProbRule = new KWDRTargetProbAt;
		targetProbRule->GetFirstOperand()->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		targetProbRule->GetFirstOperand()->SetAttributeName(classifierAttribute->GetName());
		targetProbRule->GetSecondOperand()->SetOrigin(KWDerivationRuleOperand::OriginConstant);
		targetProbRule->GetSecondOperand()->SetSymbolConstant(targetValues->GetValueAt(nTarget));

		// Ajout de l'attribut dans la classe
		targetProbAttribute = trainedClassifier->CreatePredictionAttribute(
			"Prob" + GetTargetAttributeName() + targetValues->GetValueAt(nTarget), targetProbRule);
		trainedClassifier->SetProbAttributeAt(nTarget, targetValues->GetValueAt(nTarget), targetProbAttribute);
	}
}

void KMPredictor::AddGlobalGravityCenters(KWClass* kwcModeling) {

	KMCluster* globalCluster = kmBestTrainedClustering->GetGlobalCluster();
	assert(globalCluster != NULL);

	const ContinuousVector& globalCentroid = globalCluster->GetModelingCentroidValues();
	assert(globalCentroid.GetSize() > 0);

	KWAttribute* attribute = kwcModeling->GetHeadAttribute();

	while (attribute != NULL) {

		if (attribute->GetConstMetaData()->IsKeyPresent(KMParameters::KM_ATTRIBUTE_LABEL)) {

			Object* o = parameters->GetKMAttributeNames().Lookup(attribute->GetName());
			assert(o != NULL);

			IntObject* ioLoadIndex = cast(IntObject*, o);
			const Continuous attributeGlobalGravity = globalCentroid.GetAt(ioLoadIndex->GetInt());

			attribute->GetMetaData()->SetDoubleValueAt(GLOBAL_GRAVITY_CENTER_LABEL, attributeGlobalGravity);
		}

		kwcModeling->GetNextAttribute(attribute);
	}

	kwcModeling->Compile();

}


bool KMPredictor::GenerateRecodingDictionary(KWDataPreparationClass* dataPreparationClass, ObjectArray* oaDataPreparationFilteredAttributes)
{
	KWClass* kwc = dataPreparationClass->GetDataPreparationClass();

	// Extraction du tableau des attributs de preparation
	ObjectArray* oaDataPreparationAttributes = dataPreparationClass->GetDataPreparationAttributes();

	// l'attribut cible participe forcement a la creation du modele
	KWAttribute* targetAttribute = dataPreparationClass->GetDataPreparationClass()->LookupAttribute(GetTargetAttributeName());
	if (targetAttribute != NULL) {
		targetAttribute->GetMetaData()->SetNoValueAt(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL);
		targetAttribute->GetMetaData()->SetNoValueAt("TargetVariable");
	}

	iClusteringVariablesNumber = 0;
	ObjectArray oaAddedAttributes;

	for (int nAttributeIndex = 0; nAttributeIndex < oaDataPreparationAttributes->GetSize(); nAttributeIndex++)
	{
		KWDataPreparationAttribute* dataPreparationAttribute = cast(KWDataPreparationAttribute*,
			oaDataPreparationAttributes->GetAt(nAttributeIndex));

		// Extraction de l'attribut natif
		KWAttribute* nativeAttribute = dataPreparationAttribute->GetNativeAttribute();

		KWAttribute* preprocessedAttribute = NULL;

		// On ne pretraite pas l'attribut s'il s'agit de l'attribut cible ou de l'attribut de prediction
		if (nativeAttribute->GetName() == GetTargetAttributeName() or (nativeAttribute->GetName().GetLength() >= 9 and nativeAttribute->GetName().Left(9) == "Predicted"))
			continue;


		// si l'attribut prepare ne figure pas dans les attributs filtres en amont (sur la valeur predictive de ses attributs sources, ou suite a une limitation du nombre d'attributs a evaluer), alors l'ignorer
		bool found = false;

		for (int i = 0; i < oaDataPreparationFilteredAttributes->GetSize(); i++)
			if (dataPreparationAttribute == cast(KWDataPreparationAttribute*, oaDataPreparationFilteredAttributes->GetAt(i)))
				found = true;

		if (not found) {
			nativeAttribute->SetUsed(false);
			nativeAttribute->SetLoaded(false);
			continue;
		}

		// si les variables sont des constantes, il est inutile de les garder (ce filtre est applique mmeme si on a coche la case Keep Null level variables)
		// retouver les stats correspondant a l'attribut
		KWAttributeStats* attributeStats = NULL;
		for (int i = 0; i < GetClassStats()->GetAttributeStats()->GetSize(); i++) {
			attributeStats = cast(KWAttributeStats*, GetClassStats()->GetAttributeStats()->GetAt(i));
			if (attributeStats->GetAttributeName() == nativeAttribute->GetName())
				break;
		}
		assert(attributeStats != NULL);

		if (attributeStats->GetDescriptiveStats()->GetValueNumber() == 1) {
			nativeAttribute->SetUsed(false);
			nativeAttribute->SetLoaded(false);
			continue;
		}

		// si demande, garder les variables a level nul (en mode supervise ET si les pre-traitements sont non supervises)
		if (GetTargetAttributeType() != KWType::None and dataPreparationAttribute->GetPreparedStats()->GetLevel() == 0.0) {

			// par defaut, en mode supervise, on ne prend pas en compte les variables a level nul :
			nativeAttribute->SetUsed(false);
			nativeAttribute->SetLoaded(false);

			// ... mais si les pretraitements sont non supervises, alors on peut choisir de garder les variables a level nul :
			if (parameters->GetKeepNulLevelVariables()) {

				if ((nativeAttribute->GetType() == KWType::Continuous and
					(parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::NoPreprocessing or
						parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::RankNormalization or
						parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::CenterReduction or
						parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::Normalization)
					)
					or
					(nativeAttribute->GetType() == KWType::Symbol and parameters->GetCategoricalPreprocessingType() == KMParameters::PreprocessingType::BasicGrouping)
					) {

					nativeAttribute->SetUsed(true);
					nativeAttribute->SetLoaded(true);
				}
			}
		}

		if (nativeAttribute->GetUsed() == false)
			continue;

		///////   application des pr�traitements

		// Cas d'un attribut Continuous
		if (nativeAttribute->GetType() == KWType::Continuous)
		{
			switch (parameters->GetContinuousPreprocessingType()) {

			case (KMParameters::PreprocessingType::AutomaticallyComputed):

				if (GetTargetAttributeType() == KWType::None) // non supervis�
					preprocessedAttribute = dataPreparationAttribute->AddPreparedRankNormalizedAttribute();
				else
					// supervis�
					dataPreparationAttribute->AddPreparedSourceConditionalInfoAttributes(&oaAddedAttributes);

				break;

			case (KMParameters::PreprocessingType::CenterReduction):
				preprocessedAttribute = dataPreparationAttribute->AddPreparedCenterReducedAttribute();
				break;

			case (KMParameters::PreprocessingType::UnusedVariable):
				nativeAttribute->SetUsed(false);
				nativeAttribute->SetLoaded(false);
				break;

			case (KMParameters::PreprocessingType::RankNormalization):
				preprocessedAttribute = dataPreparationAttribute->AddPreparedRankNormalizedAttribute();
				break;

			case (KMParameters::PreprocessingType::Binarization):
				dataPreparationAttribute->AddPreparedBinarizationAttributes(&oaAddedAttributes);// binarisation standard
				break;

			case (KMParameters::PreprocessingType::HammingConditionalInfo):
				AddHammingConditionalInfoAttributes(&oaAddedAttributes, dataPreparationAttribute);
				break;

			case (KMParameters::PreprocessingType::Normalization):
				preprocessedAttribute = dataPreparationAttribute->AddPreparedNormalizedAttribute();
				break;

			case (KMParameters::PreprocessingType::NoPreprocessing):
				preprocessedAttribute = nativeAttribute;
				break;

			case (KMParameters::PreprocessingType::ConditionaInfoWithPriors):
				AddConditionalInfoWithPriorsAttributes(&oaAddedAttributes, dataPreparationAttribute);
				break;

			case (KMParameters::PreprocessingType::Entropy):
				AddEntropyAttributes(&oaAddedAttributes, dataPreparationAttribute);
				break;

			case (KMParameters::PreprocessingType::EntropyWithPriors):
				AddEntropyWithPriorsAttributes(&oaAddedAttributes, dataPreparationAttribute);
				break;

			default:
				break;

			}
		}

		// Cas d'un attribut Categoriel
		if (nativeAttribute->GetType() == KWType::Symbol)
		{
			switch (parameters->GetCategoricalPreprocessingType()) {

			case (KMParameters::PreprocessingType::AutomaticallyComputed):

				if (GetTargetAttributeType() == KWType::None) // non supervis�
					// appliquer une "binarization customisee", qui renvoie 0.5 au lieu de 1, sur le resultat d'un basic grouping (NB. le basic grouping a d�j� �t� fait auparavant)
					AddPreparedBinarizationAttributes(&oaAddedAttributes, dataPreparationAttribute);
				else
					// supervise
					dataPreparationAttribute->AddPreparedSourceConditionalInfoAttributes(&oaAddedAttributes);

				break;

			case (KMParameters::PreprocessingType::UnusedVariable):
				nativeAttribute->SetUsed(false);
				nativeAttribute->SetLoaded(false);
				break;

			case (KMParameters::PreprocessingType::Binarization):
				dataPreparationAttribute->AddPreparedBinarizationAttributes(&oaAddedAttributes);// binarisation standard
				break;

			case (KMParameters::PreprocessingType::HammingConditionalInfo):
				AddHammingConditionalInfoAttributes(&oaAddedAttributes, dataPreparationAttribute);
				break;

			case (KMParameters::PreprocessingType::BasicGrouping):
				// appliquer une "binarization customisee", qui renvoie 0.5 au lieu de 1, sur le resultat d'un basic grouping (NB. le basic grouping a d�j� �t� fait auparavant)
				AddPreparedBinarizationAttributes(&oaAddedAttributes, dataPreparationAttribute);
				GetPreprocessingSpec()->GetGrouperSpec()->SetSupervisedMethodName("Grouping +  binarization");
				break;

			case (KMParameters::PreprocessingType::ConditionaInfoWithPriors):
				AddConditionalInfoWithPriorsAttributes(&oaAddedAttributes, dataPreparationAttribute);
				break;

			case (KMParameters::PreprocessingType::Entropy):
				AddEntropyAttributes(&oaAddedAttributes, dataPreparationAttribute);
				break;

			case (KMParameters::PreprocessingType::EntropyWithPriors):
				AddEntropyWithPriorsAttributes(&oaAddedAttributes, dataPreparationAttribute);
				break;

			default:
				break;

			}
		}
		// ajouter les metadata necessaires au traitement
		AddAttributesMetaData(nativeAttribute, preprocessedAttribute, oaAddedAttributes);

		// en supervis� et mode "auto-auto", ajout des attributs CellIndex, necessaires au calcul des levels de clustering figurant dans le ModelingReport
		if (GetTargetAttributeName() != "" and
			parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed and
			parameters->GetCategoricalPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed and
			dataPreparationAttribute->GetPreparedAttribute()->GetStructureName() == "DataGrid" and
			dataPreparationAttribute->GetPreparedAttribute()->GetConstMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()) > 0) {

			AddCellIndexAttribute(kwc, dataPreparationAttribute->GetPreparedAttribute(), nativeAttribute);
		}

		if (parameters->GetWriteDetailedStatistics())
			// pouvoir retrouver facilement le nom d'attribut natif d'un attribut pr�par�, lors d'une �valuation de mod�le
			dataPreparationAttribute->GetPreparedAttribute()->GetMetaData()->SetStringValueAt(PREPARED_ATTRIBUTE_METADATA, nativeAttribute->GetName());

		oaAddedAttributes.RemoveAll();
	}

	// ne garder en memoire que les attributs utiles
	if (not PrepareLearningClass(kwc, targetAttribute))
		return false;

	if (parameters->GetRecodedAttributesNames().GetCount() == 0) {
		AddWarning("No attribute has been selected for clustering processing.");
		return false;
	}

	// preparer la lecture de la base a l'aide du dico contenant les variables pre-traitees
	GetDatabase()->SetClassName(kwc->GetName());
	KWClassDomain::SetCurrentDomain(kwc->GetDomain());

	// renseigner les differentes structures permettant de retrouver les informations li�es aux attributs
	parameters->AddAttributes(kwc);

	return true;
}

boolean KMPredictor::GenerateBaselineModelingDictionary(KWTrainedClassifier* trainedClassifier,
	KWDataPreparationClass* dataPreparationClass,
	ObjectArray* oaDataPreparationUsedAttributes)
{
	KWAttribute* classifierAttribute;
	KWAttribute* targetValuesAttribute;

	require(trainedClassifier != NULL);
	require(trainedClassifier->GetPredictorClass() != NULL);
	require(dataPreparationClass != NULL);
	require(oaDataPreparationUsedAttributes != NULL);

	// Memorisation de la reference a l'attribut cible
	trainedClassifier->SetTargetAttribute(
		trainedClassifier->GetPredictorClass()->LookupAttribute(GetTargetAttributeName()));

	// Recherche de l'attribut memorisant les valeurs cibles
	targetValuesAttribute = dataPreparationClass->GetDataPreparationTargetAttribute()->GetPreparedAttribute();
	trainedClassifier->SetTargetValuesAttribute(targetValuesAttribute);


	// Ajout de l'attribut de prediction
	classifierAttribute = CreateBaselineModelClassifierAttribute(trainedClassifier, oaDataPreparationUsedAttributes);

	// Ajout des attributs de prediction pour la classification
	AddClassifierPredictionAttributes(trainedClassifier, classifierAttribute);

	// Completion automatique des informations de la classe (nom de classe par regle...)
	trainedClassifier->GetPredictorClass()->CompleteTypeInfo();

	TaskProgression::DisplayProgression(100);

	return true;
}

boolean KMPredictor::PrepareLearningClass(KWClass* kwc, KWAttribute* targetAttribute) {

	// ne garder en memoire que les attributs qui seront necessaires :
	// outre les attributs necessaires a la convergence KMean, on doit aussi garder les attributs natifs selectionnes (afin de pouvoir ecrire
	// les centroides natifs dans le preparationReport) et les attributs cellIndex (afin de pouvoir ecrire
	// les levels de clustering dans le preparationReport)

	KWAttribute* attribute = kwc->GetHeadAttribute();
	while (attribute != NULL)
	{
		if (attribute != targetAttribute and
			not attribute->GetConstMetaData()->IsKeyPresent(KMParameters::KM_ATTRIBUTE_LABEL) and
			not attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::CELL_INDEX_METADATA) and
			not attribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL)) {
			attribute->SetUsed(false);
			attribute->SetLoaded(false);
		}
		kwc->GetNextAttribute(attribute);
	}
	if (not kwc->Check())
		return false;

	kwc->Compile();
	return true;
}

void KMPredictor::CreatePredictorReport()
{
	require(bIsTraining);
	require(predictorReport == NULL);

	predictorReport = new KMPredictorReport;
	predictorReport->SetLearningSpec(GetLearningSpec());
	predictorReport->SetPredictorName(GetName());
}

KWPredictorEvaluation* KMPredictor::Evaluate(KWDatabase* database)
{
	require(IsTrained());
	require(database != NULL);
	require(kmBestTrainedClustering != NULL);

	Global::SetSilentMode(false);

	if (GetTargetAttributeType() == KWType::Symbol)
	{
		// mode supervis�
		KMClassifierEvaluation* classifierEvaluation = new KMClassifierEvaluation;
		classifierEvaluation->Evaluate(this, database);
		return classifierEvaluation;
	}
	else
	{
		// mode non supervis�
		KMPredictorEvaluation* predictorEvaluation = new KMPredictorEvaluation;
		predictorEvaluation->Evaluate(this, database);
		return predictorEvaluation;
	}
}

boolean KMPredictor::HasSufficientMemoryForTraining(KWDataPreparationClass* dataPreparationClass, const int nInstancesNumber)
{
	const double dAvailableMemory = RMResourceManager::GetRemainingAvailableMemory();

	double dWantedMemory = ComputeRequiredMemory(nInstancesNumber, dataPreparationClass->GetDataPreparationClass());

	if (parameters->GetVerboseMode() and dAvailableMemory < dWantedMemory) {
		std::stringstream ss;
		ss << std::fixed << "Available memory = " << dAvailableMemory / 1024 / 1024 <<
			" Mo, needed memory for training phase = " << dWantedMemory / 1024 / 1024 << " Mo.";
		AddWarning(ALString(ss.str().c_str()));

#ifdef WIN32
		AddMessage("For higher memory ressources, you may try to use the 64 bits version of MLClusters.");
#endif
	}

	return (dAvailableMemory >= dWantedMemory ? true : false);
}

/** evaluer grossierement la memoire necessaire pour traiter n instances en base de donnees, a partir du dico fourni en entr�e */
longint KMPredictor::ComputeRequiredMemory(const longint instancesNumber, const KWClass* kwc) {

	// Memoire pour charger 1 attribut de la base (repris de la methode LearningEnv : KWClassStats::ComputeMaxLoadableAttributeNumber() )
	double dDatabaseAttributeMemory = (1 + instancesNumber) * 1.0 * sizeof(KWValue);

	longint dRequiredMemory = dDatabaseAttributeMemory * kwc->GetLoadedAttributeNumber();
	dRequiredMemory += kwc->GetUsedMemory();
	dRequiredMemory += (instancesNumber * sizeof(KWObject*));

	return dRequiredMemory;
}

void KMPredictor::AddAttributesMetaData(KWAttribute* nativeAttribute, KWAttribute* preprocessedAttribute,
	ObjectArray& oaAddedAttributes) {

	// ajouter les metadata necessaires au traitement

	assert(nativeAttribute != NULL);

	if (nativeAttribute->GetUsed() and nativeAttribute->GetLoaded())
		// reperer les attibuts natifs ayant participe a la creation du modele (i.e, qui n'ont pas ete deselectionnes via l'IHM avant le lancement de l'apprentissage)
		nativeAttribute->GetMetaData()->SetNoValueAt(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL);

	if (preprocessedAttribute != NULL) {

		if (nativeAttribute->GetUsed())
			preprocessedAttribute->GetMetaData()->SetDoubleValueAt(KMParameters::KM_ATTRIBUTE_LABEL, ++iClusteringVariablesNumber);

		if (GetTargetAttributeName() != "") {

			// prendre en compte le cas ou la variable n'est pas pretraitee
			if (nativeAttribute != preprocessedAttribute)
				nativeAttribute->GetMetaData()->SetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey(),
					preprocessedAttribute->GetMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()));// repercuter le level de l'attribut pretrait� sur l'attribut natif
		}

		parameters->AddRecodedAttribute(nativeAttribute, preprocessedAttribute);
	}

	for (int i = 0; i < oaAddedAttributes.GetSize(); i++) {

		KWAttribute* attribute = cast(KWAttribute*, oaAddedAttributes.GetAt(i));

		if (nativeAttribute->GetUsed())
			attribute->GetMetaData()->SetDoubleValueAt(KMParameters::KM_ATTRIBUTE_LABEL, ++iClusteringVariablesNumber);

		if (GetTargetAttributeName() != "") {
			nativeAttribute->GetMetaData()->SetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey(),
				attribute->GetMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()));// repercuter le level de l'attribut pretrait� sur l'attribut natif
		}

		// tableau de correspondance entre attributs natifs et recod�s
		parameters->AddRecodedAttribute(nativeAttribute, attribute);
	}
}

// customisation de la binarisation d'attributs : au lieu de renvoyer 0 ou 1, la formule de derivation renvoie 0 ou 0.5
void KMPredictor::AddPreparedBinarizationAttributes(ObjectArray* oaAddedAttributes, KWDataPreparationAttribute* dataPreparationAttribute)
{
	KWAttribute* cellIndexAttribute;
	KWAttribute* binaryAttribute;
	const ALString sBinaryPrefix = "B";
	int nBinaryAttributeNumber;
	int i;

	// Creation d'un attribut intermediaire (Unused) pour le calcul de l'index
	cellIndexAttribute = dataPreparationAttribute->AddPreparedIndexingAttribute();
	cellIndexAttribute->SetUsed(false);

	// Parcours des cellules sources la grille en supervise, ou cibles en non supervise
	if (dataPreparationAttribute->GetPreparedStats()->GetTargetAttributeType() == KWType::None)
	{
		assert(dataPreparationAttribute->GetPreparedStats()->GetPreparedDataGridStats()->ComputeSourceGridSize() <= 1);
		nBinaryAttributeNumber = dataPreparationAttribute->GetPreparedStats()->GetPreparedDataGridStats()->ComputeTargetGridSize();
	}
	else
		nBinaryAttributeNumber = dataPreparationAttribute->GetPreparedStats()->GetPreparedDataGridStats()->ComputeSourceGridSize();

	oaAddedAttributes->SetSize(0);

	for (i = 0; i < nBinaryAttributeNumber; i++)
	{
		// Creation d'une regle de calcul du type PRODUCT(EQ(IndexPworkclass, 2), 0.5))
		KWDREQ* eqRule = new KWDREQ;
		eqRule->GetFirstOperand()->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		eqRule->GetFirstOperand()->SetAttributeName(cellIndexAttribute->GetName());
		eqRule->GetSecondOperand()->SetOrigin(KWDerivationRuleOperand::OriginConstant);
		eqRule->GetSecondOperand()->SetContinuousConstant((Continuous)(i + 1));

		KWDRProduct* productRule = new KWDRProduct;
		productRule->DeleteAllOperands();
		KWDerivationRuleOperand* productOperand1 = new KWDerivationRuleOperand;
		productOperand1->SetOrigin(KWDerivationRuleOperand::OriginRule);
		productOperand1->SetDerivationRule(eqRule);
		KWDerivationRuleOperand* productOperand2 = new KWDerivationRuleOperand;
		productOperand2->SetOrigin(KWDerivationRuleOperand::OriginConstant);
		productOperand2->SetType(KWType::Continuous);
		productOperand2->SetContinuousConstant(0.5);

		productRule->AddOperand(productOperand1);
		productRule->AddOperand(productOperand2);

		// Creation de l'attribut de binarisation
		binaryAttribute = AddDataPreparationRuleAttribute(productRule, sBinaryPrefix + IntToString(i + 1), dataPreparationAttribute);
		oaAddedAttributes->Add(binaryAttribute);
	}
}

void KMPredictor::AddHammingConditionalInfoAttributes(ObjectArray* oaAddedAttributes, KWDataPreparationAttribute* dataPreparationAttribute)
{
	/* pour chaque attribut natif, g�n�rer des attributs HC (Hamming Conditionalinfo) sur le mod�le ci-dessous (en prenant pour exemple uniquement SepalLength) :

	Numerical	`HC1_Iris-setosa_PSepalLength`	 = Product(EQ(IndexPSepalLength, 1), SourceConditionalInfo(StatsPSepalLength, 1))	; <KmeansAttribute=1> <Level=0.293137>
	Numerical	`HC1_Iris-versicolor_PSepalLength`	 = Product(EQ(IndexPSepalLength, 1), SourceConditionalInfo(StatsPSepalLength, 2))	; <KmeansAttribute=2> <Level=0.293137>
	Numerical	`HC1_Iris-virginica_PSepalLength`	 = Product(EQ(IndexPSepalLength, 1), SourceConditionalInfo(StatsPSepalLength, 3))	; <KmeansAttribute=3> <Level=0.293137>
	Numerical	`HC2_Iris-setosa_PSepalLength`	 = Product(EQ(IndexPSepalLength, 2), SourceConditionalInfo(StatsPSepalLength, 1))	; <KmeansAttribute=4> <Level=0.293137>
	Numerical	`HC2_Iris-versicolor_PSepalLength`	 = Product(EQ(IndexPSepalLength, 2), SourceConditionalInfo(StatsPSepalLength, 2))	; <KmeansAttribute=5> <Level=0.293137>
	Numerical	`HC2_Iris-virginica_PSepalLength`	 = Product(EQ(IndexPSepalLength, 2), SourceConditionalInfo(StatsPSepalLength, 3))	; <KmeansAttribute=6> <Level=0.293137>
	Numerical	`HC3_Iris-setosa_PSepalLength`	 = Product(EQ(IndexPSepalLength, 3), SourceConditionalInfo(StatsPSepalLength, 1))	; <KmeansAttribute=7> <Level=0.293137>
	Numerical	`HC3_Iris-versicolor_PSepalLength`	 = Product(EQ(IndexPSepalLength, 3), SourceConditionalInfo(StatsPSepalLength, 2))	; <KmeansAttribute=8> <Level=0.293137>
	Numerical	`HC3_Iris-virginica_PSepalLength`	 = Product(EQ(IndexPSepalLength, 3), SourceConditionalInfo(StatsPSepalLength, 3))	; <KmeansAttribute=9> <Level=0.293137>

	*/

	ObjectArray binarizationAttributes;
	dataPreparationAttribute->AddPreparedBinarizationAttributes(&binarizationAttributes);

	ObjectArray conditionalInfoAttributes;
	dataPreparationAttribute->AddPreparedSourceConditionalInfoAttributes(&conditionalInfoAttributes);
	assert(conditionalInfoAttributes.GetSize() > 0);
	KWAttribute* ciAttribute = cast(KWAttribute*, conditionalInfoAttributes.GetAt(0));
	KWAttribute* statsAttribute = ciAttribute->GetDerivationRule()->GetFirstOperand()->GetOriginAttribute();
	assert(statsAttribute != NULL);

	const int nIntervalsNumber = binarizationAttributes.GetSize();

	KWAttribute* indexingAttribute = dataPreparationAttribute->GetPreparedAttribute()->GetParentClass()->LookupAttribute("Index" + dataPreparationAttribute->GetPreparedAttribute()->GetName());
	assert(indexingAttribute != NULL);

	for (int idxInterval = 0; idxInterval < nIntervalsNumber; idxInterval++) {

		const KWDGSAttributeSymbolValues* targetValues = cast(const KWDGSAttributeSymbolValues*, GetTargetValueStats()->GetAttributeAt(0));

		for (int idxModality = 0; idxModality < targetValues->GetValueNumber(); idxModality++) {

			KWDREQ* eqRule = new KWDREQ;
			eqRule->DeleteAllOperands();
			KWDerivationRuleOperand* indexingAttributeOperand = new KWDerivationRuleOperand;
			indexingAttributeOperand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
			indexingAttributeOperand->SetType(KWType::Continuous);
			indexingAttributeOperand->SetAttributeName(indexingAttribute->GetName());
			eqRule->AddOperand(indexingAttributeOperand);

			KWDerivationRuleOperand* intervalOperand = new KWDerivationRuleOperand;
			intervalOperand->SetOrigin(KWDerivationRuleOperand::OriginConstant);
			intervalOperand->SetType(KWType::Continuous);
			intervalOperand->SetContinuousConstant(idxInterval + 1);
			eqRule->AddOperand(intervalOperand);

			KWDRSourceConditionalInfo* ciRule = new KWDRSourceConditionalInfo;
			ciRule->DeleteAllOperands();
			KWDerivationRuleOperand* statsAttributeOperand = new KWDerivationRuleOperand;
			statsAttributeOperand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
			statsAttributeOperand->SetType(KWType::Continuous);
			statsAttributeOperand->SetAttributeName(statsAttribute->GetName());
			ciRule->AddOperand(statsAttributeOperand);

			KWDerivationRuleOperand* indexModalityOperand = new KWDerivationRuleOperand;
			indexModalityOperand->SetOrigin(KWDerivationRuleOperand::OriginConstant);
			indexModalityOperand->SetType(KWType::Continuous);
			indexModalityOperand->SetContinuousConstant(idxModality + 1);
			ciRule->AddOperand(indexModalityOperand);

			KWDRProduct* productRule = new KWDRProduct;
			productRule->DeleteAllOperands();
			KWDerivationRuleOperand* eqOperand = new KWDerivationRuleOperand;
			eqOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
			eqOperand->SetType(KWType::Continuous);
			eqOperand->SetDerivationRule(eqRule);

			KWDerivationRuleOperand* ciOperand = new KWDerivationRuleOperand;
			ciOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
			ciOperand->SetType(KWType::Continuous);
			ciOperand->SetDerivationRule(ciRule);

			productRule->AddOperand(eqOperand);
			productRule->AddOperand(ciOperand);

			KWAttribute* hammingAttribute = AddDataPreparationRuleAttribute(productRule,
				ALString("HC") + IntToString(idxInterval + 1) + "_" +
				targetValues->GetValueAt(idxModality) + "_", dataPreparationAttribute);

			oaAddedAttributes->Add(hammingAttribute);

		}
	}
}


void KMPredictor::AddConditionalInfoWithPriorsAttributes(ObjectArray* oaAddedAttributes, KWDataPreparationAttribute* dataPreparationAttribute)
{
	/* pour chaque attribut natif, g�n�rer des attributs CIP (ConditionalInfo with prior) sur le mod�le ci-dessous :

	Unused	Numerical	CIP_Info1Page	 = Product(Info1Page, <proba de la valeur cible>)	;

	Ce qui revient a multiplier les ConditionalInfo par la proba de la modalite cible.
	*/

	ObjectArray conditionalInfoAttributes;
	dataPreparationAttribute->AddPreparedSourceConditionalInfoAttributes(&conditionalInfoAttributes);
	assert(conditionalInfoAttributes.GetSize() > 0);

	// frequences des modalites cibles :
	IntVector ivPartFrequencies;
	GetTargetValueStats()->ExportAttributePartFrequenciesAt(0, &ivPartFrequencies);

	double nTotalFrequency = 0;
	for (int nPart = 0; nPart < ivPartFrequencies.GetSize(); nPart++)
		nTotalFrequency += ivPartFrequencies.GetAt(nPart);

	assert(conditionalInfoAttributes.GetSize() == ivPartFrequencies.GetSize());

	for (int i = 0; i < conditionalInfoAttributes.GetSize(); i++) {

		KWAttribute* ciAttribute = cast(KWAttribute*, conditionalInfoAttributes.GetAt(i));

		KWDRProduct* productRule = new KWDRProduct;
		productRule->DeleteAllOperands();

		KWDerivationRuleOperand* targetProbaOperand = new KWDerivationRuleOperand;
		targetProbaOperand->SetOrigin(KWDerivationRuleOperand::OriginConstant);
		targetProbaOperand->SetType(KWType::Continuous);
		targetProbaOperand->SetContinuousConstant(((double)ivPartFrequencies.GetAt(i)) / nTotalFrequency);

		KWDerivationRuleOperand* ciOperand = new KWDerivationRuleOperand;
		ciOperand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		ciOperand->SetAttributeName(ciAttribute->GetName());

		productRule->AddOperand(targetProbaOperand);
		productRule->AddOperand(ciOperand);

		KWAttribute* cipAttribute = AddDataPreparationRuleAttribute(productRule, ALString("CIP") + ALString(IntToString(i + 1)) + "_", dataPreparationAttribute);
		oaAddedAttributes->Add(cipAttribute);
	}
}


void KMPredictor::AddEntropyAttributes(ObjectArray* oaAddedAttributes, KWDataPreparationAttribute* dataPreparationAttribute)
{

	/* pour chaque attribut natif, g�n�rer des attributs EN (entropy) sur le mod�le ci-dessous :

	Unused	Numerical	EN_Info1Page	 = Product(Info1Page, Exp(Product(-1, Info1Page)))	;

	*/

	ObjectArray conditionalInfoAttributes;
	dataPreparationAttribute->AddPreparedSourceConditionalInfoAttributes(&conditionalInfoAttributes);
	assert(conditionalInfoAttributes.GetSize() > 0);

	// frequences des modalites cibles :
	IntVector ivPartFrequencies;
	GetTargetValueStats()->ExportAttributePartFrequenciesAt(0, &ivPartFrequencies);

	double nTotalFrequency = 0;
	for (int nPart = 0; nPart < ivPartFrequencies.GetSize(); nPart++)
		nTotalFrequency += ivPartFrequencies.GetAt(nPart);

	assert(conditionalInfoAttributes.GetSize() == ivPartFrequencies.GetSize());

	for (int i = 0; i < conditionalInfoAttributes.GetSize(); i++) {

		KWAttribute* ciAttribute = cast(KWAttribute*, conditionalInfoAttributes.GetAt(i));

		// -1
		KWDerivationRuleOperand* minusOperand = new KWDerivationRuleOperand;
		minusOperand->SetOrigin(KWDerivationRuleOperand::OriginConstant);
		minusOperand->SetType(KWType::Continuous);
		minusOperand->SetContinuousConstant(-1);

		// Info1Page
		KWDerivationRuleOperand* ciOperand = new KWDerivationRuleOperand;
		ciOperand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		ciOperand->SetAttributeName(ciAttribute->GetName());

		// Product(-1, Info1Page)
		KWDRProduct* productConditionalInfoRule = new KWDRProduct;
		productConditionalInfoRule->DeleteAllOperands();
		productConditionalInfoRule->AddOperand(minusOperand);
		productConditionalInfoRule->AddOperand(ciOperand);

		// Exp(Product(-1, Info1Page))
		KWDRExp* expRule = new KWDRExp;
		expRule->DeleteAllOperands();
		KWDerivationRuleOperand* expRuleOperand = new KWDerivationRuleOperand;
		expRuleOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
		expRuleOperand->SetDerivationRule(productConditionalInfoRule);
		expRule->AddOperand(expRuleOperand);

		// Info1Page
		KWDerivationRuleOperand* ciOperand2 = new KWDerivationRuleOperand;
		ciOperand2->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		ciOperand2->SetAttributeName(ciAttribute->GetName());

		// Product(Info1Page, Exp(Product(-1, Info1Page)))
		KWDerivationRuleOperand* expOperand = new KWDerivationRuleOperand;
		expOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
		expOperand->SetDerivationRule(expRule);
		KWDRProduct* productRule = new KWDRProduct;
		productRule->DeleteAllOperands();
		productRule->AddOperand(ciOperand2);
		productRule->AddOperand(expOperand);

		KWAttribute* cipAttribute = AddDataPreparationRuleAttribute(productRule, ALString("EN") + ALString(IntToString(i + 1)) + "_", dataPreparationAttribute);
		oaAddedAttributes->Add(cipAttribute);
	}
}


void KMPredictor::AddEntropyWithPriorsAttributes(ObjectArray* oaAddedAttributes, KWDataPreparationAttribute* dataPreparationAttribute)
{

	/* pour chaque attribut natif, g�n�rer des attributs EN (entropy) sur le mod�le ci-dessous :

	Unused	Numerical	ENP_Info1Page	 = Product(Info1Page, Exp(Product(-1, Info1Page)), <proba de la valeur cible>)	;

	*/

	ObjectArray conditionalInfoAttributes;
	dataPreparationAttribute->AddPreparedSourceConditionalInfoAttributes(&conditionalInfoAttributes);
	assert(conditionalInfoAttributes.GetSize() > 0);

	// frequences des modalites cibles :
	IntVector ivPartFrequencies;
	GetTargetValueStats()->ExportAttributePartFrequenciesAt(0, &ivPartFrequencies);

	double nTotalFrequency = 0;
	for (int nPart = 0; nPart < ivPartFrequencies.GetSize(); nPart++)
		nTotalFrequency += ivPartFrequencies.GetAt(nPart);

	assert(conditionalInfoAttributes.GetSize() == ivPartFrequencies.GetSize());

	for (int i = 0; i < conditionalInfoAttributes.GetSize(); i++) {

		KWAttribute* ciAttribute = cast(KWAttribute*, conditionalInfoAttributes.GetAt(i));

		// -1
		KWDerivationRuleOperand* minusOperand = new KWDerivationRuleOperand;
		minusOperand->SetOrigin(KWDerivationRuleOperand::OriginConstant);
		minusOperand->SetType(KWType::Continuous);
		minusOperand->SetContinuousConstant(-1);

		// Info1Page
		KWDerivationRuleOperand* ciOperand = new KWDerivationRuleOperand;
		ciOperand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		ciOperand->SetAttributeName(ciAttribute->GetName());

		// Product(-1, Info1Page)
		KWDRProduct* productConditionalInfoRule = new KWDRProduct;
		productConditionalInfoRule->DeleteAllOperands();
		productConditionalInfoRule->AddOperand(minusOperand);
		productConditionalInfoRule->AddOperand(ciOperand);

		// Exp(Product(-1, Info1Page))
		KWDRExp* expRule = new KWDRExp;
		expRule->DeleteAllOperands();
		KWDerivationRuleOperand* expRuleOperand = new KWDerivationRuleOperand;
		expRuleOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
		expRuleOperand->SetDerivationRule(productConditionalInfoRule);
		expRule->AddOperand(expRuleOperand);

		// Info1Page
		KWDerivationRuleOperand* ciOperand2 = new KWDerivationRuleOperand;
		ciOperand2->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
		ciOperand2->SetAttributeName(ciAttribute->GetName());

		// <proba de la valeur cible>
		KWDerivationRuleOperand* targetProbaOperand = new KWDerivationRuleOperand;
		targetProbaOperand->SetOrigin(KWDerivationRuleOperand::OriginConstant);
		targetProbaOperand->SetType(KWType::Continuous);
		targetProbaOperand->SetContinuousConstant(((double)ivPartFrequencies.GetAt(i)) / nTotalFrequency);

		// Product(Info1Page, Exp(Product(-1, Info1Page)), <proba de la valeur cible>)
		KWDerivationRuleOperand* expOperand = new KWDerivationRuleOperand;
		expOperand->SetOrigin(KWDerivationRuleOperand::OriginRule);
		expOperand->SetDerivationRule(expRule);
		KWDRProduct* productRule = new KWDRProduct;
		productRule->DeleteAllOperands();
		productRule->AddOperand(ciOperand2);
		productRule->AddOperand(expOperand);
		productRule->AddOperand(targetProbaOperand);

		KWAttribute* cipAttribute = AddDataPreparationRuleAttribute(productRule, ALString("ENP") + ALString(IntToString(i + 1)) + "_", dataPreparationAttribute);
		oaAddedAttributes->Add(cipAttribute);
	}
}

KWAttribute* KMPredictor::AddDataPreparationRuleAttribute(KWDerivationRule* preparationRule, const ALString& sAttributePrefix, KWDataPreparationAttribute* dataPreparationAttribute)
{
	KWClass* kwcDataPreparationClass;
	KWAttribute* dataGridRuleAttribute;

	require(Check());
	require(preparationRule != NULL);

	// Acces a la classe de preparation
	kwcDataPreparationClass = dataPreparationAttribute->GetPreparedAttribute()->GetParentClass();

	// Ajout de l'attribut exploitant la regle de preparation
	dataGridRuleAttribute = new KWAttribute;
	dataGridRuleAttribute->SetName(sAttributePrefix + dataPreparationAttribute->GetPreparedAttribute()->GetName());
	dataGridRuleAttribute->SetDerivationRule(preparationRule);

	// Completion des informations pour finaliser la specification
	dataGridRuleAttribute->CompleteTypeInfo(kwcDataPreparationClass);

	// Ajout d'un libelle indiquant le ProbLevel
	dataGridRuleAttribute->GetMetaData()->SetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey(), dataPreparationAttribute->GetPreparedStats()->GetLevel());

	// Ajout dans la classe
	dataGridRuleAttribute->SetName(kwcDataPreparationClass->BuildAttributeName(dataGridRuleAttribute->GetName()));
	kwcDataPreparationClass->InsertAttribute(dataGridRuleAttribute);
	return dataGridRuleAttribute;
}

void KMPredictor::ExtractSourceConditionalInfoCategorical(const KWAttribute* attribute, const KWAttribute* nativeAttribute, KWClass* kwc) {

	assert(attribute->GetLoaded() and attribute->GetUsed());

	// on a une ligne de dico comme : Continuous CellIndexVClass	 = CellIndex(VClass, Class)

	ALString originalAttributeName = attribute->GetDerivationRule()->GetFirstOperand()->GetAttributeName();
	KWAttribute* originalAttribute = kwc->LookupAttribute(originalAttributeName);
	assert(originalAttribute != NULL);

	// ne pas prendre en compte les attributs a level nul
	if (nativeAttribute->GetConstMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()) == 0)
		return;

	// recuperer les groupes
	if (originalAttribute->GetDerivationRule()->GetFirstOperand()->GetDerivationRule()->GetName() == "ValueGroups") { // attention, la variable cible a une autre regle de derivation

		KWDRValueGroups* kwdrGroups = cast(KWDRValueGroups*,
			originalAttribute->GetDerivationRule()->GetFirstOperand()->GetDerivationRule());

		kmBestTrainedClustering->GetAttributesPartitioningManager()->AddValueGroups(kwdrGroups, attribute->GetName(), 3, (GetTargetAttributeName() == "" ? false : true));
	}
}

void KMPredictor::ExtractSourceConditionalInfoContinuous(const KWAttribute* attribute, const KWAttribute* nativeAttribute, KWClass* kwc) {

	assert(attribute->GetLoaded() and attribute->GetUsed());

	// on a une ligne de type :
	// Continuous	CellIndexPSepalLength	 = CellIndex(PSepalLength, SepalLength)	;

	ALString originalAttributeName = attribute->GetDerivationRule()->GetFirstOperand()->GetAttributeName();
	KWAttribute* originalAttribute = kwc->LookupAttribute(originalAttributeName);
	assert(originalAttribute != NULL);

	// ne pas prendre en compte les attributs a level nul
	if (nativeAttribute->GetConstMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()) == 0)
		return;

	// originalAttribute est de la forme : Structure(DataGrid)	PSepalLength = DataGrid(IntervalBounds(5.45, 6.15), SymbolValueSet("Iris-setosa", "Iris-versicolor", "Iris-virginica"), Frequencies(45, 5, 0, 6, 28, 16, 1, 10, 39))
	KWDRIntervalBounds* kwdrIntervalBounds = cast(KWDRIntervalBounds*,
		originalAttribute->GetDerivationRule()->GetFirstOperand()->GetDerivationRule());

	kmBestTrainedClustering->GetAttributesPartitioningManager()->AddIntervalBounds(kwdrIntervalBounds, attribute->GetName());

}

const char* KMPredictor::ID_CLUSTER_METADATA = "ClusterIdAttribute";
const char* KMPredictor::CELL_INDEX_METADATA = "KmeanCellIndex";
const char* KMPredictor::PREPARED_ATTRIBUTE_METADATA = "PreparedAttribute";
const char* KMPredictor::PREDICTOR_NAME = "KMean";
const char* KMPredictor::DISTANCE_CLUSTER_LABEL = "DistanceCluster";
const char* KMPredictor::CLUSTER_LABEL = "ClusterLabel";
const char* KMPredictor::ID_CLUSTER_LABEL = "IdCluster";
const char* KMPredictor::GLOBAL_GRAVITY_CENTER_LABEL = "GlobalGravityCenter";


// ===========  methodes globales (tri)

int
KMCompareLevel(const void* elem1, const void* elem2)
{
	KWAttribute* attr1 = (KWAttribute*) * (Object**)elem1;
	KWAttribute* attr2 = (KWAttribute*) * (Object**)elem2;

	// Comparaison de 2 attributs sur le Level
	const double level1 = attr1->GetMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey());
	const double level2 = attr2->GetMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey());

	return (level1 > level2 ? -1 : 1);
}


int
KMCompareAttributeName(const void* elem1, const void* elem2)
{
	KWAttribute* attr1 = (KWAttribute*) * (Object**)elem1;
	KWAttribute* attr2 = (KWAttribute*) * (Object**)elem2;

	// Comparaison de 2 attributs sur leur nom
	const ALString s1 = attr1->GetName();
	const ALString s2 = attr2->GetName();

#ifndef __UNIX__

	int i = _stricmp(s1, s2);

	if (i == 0)
		return 0;
	else
		if (i > 0)
			return 1;
		else
			return -1;
#else

	int i = strcasecmp(s1, s2);

	if (i == 0)
		return 0;
	else
		if (i > 0)
			return 1;
		else
			return -1;
#endif

}










