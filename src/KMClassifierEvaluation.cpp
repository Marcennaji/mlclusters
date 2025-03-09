// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMClassifierEvaluation.h"
#include "KMClusteringQuality.h"
#include "KMClassifierEvaluationTask.h"

#include <sstream>

int KMCompareTargetProbs(const void* elem1, const void* elem2);

KMClassifierEvaluation::KMClassifierEvaluation() {

	trainedPredictor = NULL;
	predictorEvaluationTask = NULL;
	lInstanceEvaluationNumber = 0;
}

KMClassifierEvaluation::~KMClassifierEvaluation() {

	if (predictorEvaluationTask != NULL)
		delete predictorEvaluationTask;
}

void KMClassifierEvaluation::WriteFullReport(ostream& ost,
	const ALString& sEvaluationLabel,
	ObjectArray* oaPredictorEvaluations)
{
	require(oaPredictorEvaluations != NULL);
	require(CheckPredictorEvaluations(oaPredictorEvaluations));
	assert(predictorEvaluationTask != NULL);
	assert(predictorEvaluationTask->GetClustering() != NULL);

	lInstanceEvaluationNumber = predictorEvaluationTask->GetInstanceEvaluationNumber();

	const ContinuousVector& globalGravity = predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetEvaluationCentroidValues();

	if (globalGravity.GetSize() == 0) {
		ost << endl << "No result. Hint : check your discard mode parameters";
	}
	else {

		// Titre et caracteristiques de la base d'evaluation
		ost << sEvaluationLabel << " ";
		ost << "evaluation report" << "\n";
		ost << "\n";
		ost << "Dictionary" << "\t" << GetClass()->GetName() << "\n";
		ost << "Target variable" << "\t" << KWType::ToString(GetTargetAttributeType()) << "\t" << GetTargetAttributeName() << "\n";
		ost << "Main target value" << "\t" << GetMainTargetModality() << "\n";
		ost << "Database\t" << GetDatabaseName() << "\n";
		ost << "Instances\t" << GetEvaluationInstanceNumber() << "\n";

		// Tableau synthetique des performances des predicteurs
		WriteArrayLineReport(ost, "Predictors performance", oaPredictorEvaluations);

		// Tableau detaille des performances des predicteurs
		WriteArrayReport(ost, "Predictors detailed performance", oaPredictorEvaluations);

		// statistiques kmean
		WriteKMeanStatistics(ost);

		// Courbes de performance
		WritePerformanceCurveReportArray(ost, oaPredictorEvaluations);

	}

	KMPredictorEvaluation::CleanPredictorClass(trainedPredictor->GetPredictorClass());
}

// ecriture du rapport d evaluation (train ou test)
void KMClassifierEvaluation::WriteKMeanStatistics(ostream& ost) {

	assert(predictorEvaluationTask != NULL);
	assert(predictorEvaluationTask->GetClustering() != NULL);
	assert(predictorEvaluationTask != NULL);
	const KMParameters* parameters = predictorEvaluationTask->GetClustering()->GetParameters();
	assert(parameters != NULL);
	assert(trainedPredictor != NULL);
	assert(trainedPredictor->GetPredictorClass() != NULL);

	ost << endl << "Evaluated instances number : " << lInstanceEvaluationNumber << endl;

	// calcul du ratio inertie inter/inertie totale
	const double totalInerty = (1.0 / (double)lInstanceEvaluationNumber) * predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetDistanceSum(parameters->GetDistanceType());
	double inertyInter = 0;
	for (int idxCluster = 0; idxCluster < predictorEvaluationTask->GetClustering()->GetClusters()->GetSize(); idxCluster++) {
		KMCluster* c = cast(KMCluster*, predictorEvaluationTask->GetClustering()->GetClusters()->GetAt(idxCluster));
		inertyInter += c->GetInertyInter(parameters->GetDistanceType());
	}

	ost << endl << "Clustering statistics : " << endl << endl
		<< "Clustering\tMean distance\tInerty inter / total\tDavies-Bouldin  (L2)\tARI by clusters\tPredictive clustering";

	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\tARI by classes";
	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\tEVA";
	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\tLEVA";
	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\tVariation of Information";
	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\tNMI by clusters";
	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\tNMI by classes";

	ost << endl << "KMean\t"
		<< ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetMeanDistance())) << "\t"
		<< ALString(DoubleToString(inertyInter / totalInerty)) << "\t"
		<< ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetDaviesBouldin())) << "\t"
		<< ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetARIByClusters())) << "\t"
		<< ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetPredictiveClustering()));

	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\t" << ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetARIByClasses()));
	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics())) {
		double d = (predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetEVA() < 0 ? 0 : predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetEVA());
		ost << "\t" << ALString(DoubleToString(d));
	}
	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\t" << ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetLEVA()));
	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\t" << ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetVariationOfInformation()));
	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\t" << ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetNormalizedMutualInformationByClusters()));
	if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()))
		ost << "\t" << ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetNormalizedMutualInformationByClasses()));

	ost << endl << endl;

	WriteClustersGravityCenters(ost);

	if (GetLearningExpertMode() and predictorEvaluationTask->GetClustering()->GetParameters()->GetWriteDetailedStatistics()) {
		KMPredictorEvaluation::WriteClustersDistancesUnnormalized(ost, predictorEvaluationTask->GetClustering());
		KMPredictorEvaluation::WriteClustersDistancesNormalized(ost, predictorEvaluationTask->GetClustering());
		KMPredictorEvaluation::WriteTrainTestCentroidsShifting(ost, predictorEvaluationTask->GetClustering());
	}

	if (parameters->GetWriteDetailedStatistics()) {

		TaskProgression::BeginTask();
		TaskProgression::SetTitle("Detailed statistics");
		TaskProgression::DisplayLabel("Writing detailed statistics...");
		TaskProgression::DisplayProgression(0);

		// tri des attributs par ordre decroissant de level (si supervisé), ou par nom (si non supervisé)
		boolean bSortOnLevel = false;
		boolean bHasNativeCategoricalAttributes = false;

		ObjectArray* oaAttributesList = new ObjectArray;

		KWAttribute* attribute = trainedPredictor->GetPredictorClass()->GetHeadAttribute();
		while (attribute != NULL)
		{
			if (parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL) {

				oaAttributesList->Add(attribute);

				if (attribute->GetConstMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()) > 0)
					bSortOnLevel = true;

				if (attribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL)) {
					if (attribute->GetType() == KWType::Symbol and not attribute->GetConstMetaData()->IsKeyPresent("TargetVariable"))
						bHasNativeCategoricalAttributes = true;
				}
			}

			trainedPredictor->GetPredictorClass()->GetNextAttribute(attribute);
		}

		oaAttributesList->SetCompareFunction(bSortOnLevel ? KMCompareLevel : KMCompareAttributeName);
		oaAttributesList->Sort();

		KMPredictorEvaluation::WriteContinuousMeanValues(ost, predictorEvaluationTask->GetClustering(), oaAttributesList);
		TaskProgression::DisplayProgression(5);
		KMPredictorEvaluation::WriteContinuousMedianValues(ost, predictorEvaluationTask->GetClustering(), oaAttributesList, predictorEvaluationTask->GetReadInstancesForMedianComputation(), lInstanceEvaluationNumber);
		TaskProgression::DisplayProgression(10);

		if (GetLearningExpertMode() and bHasNativeCategoricalAttributes) {
			KMPredictorEvaluation::WriteCategoricalModeValues(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetAtomicModalitiesFrequencyTables(), oaAttributesList, trainedPredictor->GetPredictorClass());
			TaskProgression::DisplayProgression(20);
			KMPredictorEvaluation::WritePercentagePerLineModeValues(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetAtomicModalitiesFrequencyTables(), oaAttributesList);
		}
		TaskProgression::DisplayProgression(50);
		KMPredictorEvaluation::WriteNativeAttributesProbs(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), oaAttributesList);
		TaskProgression::DisplayProgression(60);
		KMPredictorEvaluation::WritePercentagePerLineNativeAttributesProbs(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), oaAttributesList);
		TaskProgression::DisplayProgression(70);

		if (GetLearningExpertMode()) {
			KMPredictorEvaluation::WriteCumulativeNativeAttributesProbs(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), true, oaAttributesList); // cumulatif ascendant
			TaskProgression::DisplayProgression(80);
			KMPredictorEvaluation::WriteCumulativeNativeAttributesProbs(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), false, oaAttributesList);// cumulatif descendant
			TaskProgression::DisplayProgression(90);
			KMPredictorEvaluation::WriteGlobalGravityCenters(ost, predictorEvaluationTask->GetClustering());
		}
		TaskProgression::DisplayProgression(100);
		TaskProgression::EndTask();

		delete oaAttributesList;
	}
}

void KMClassifierEvaluation::WriteClustersGravityCenters(ostream& ost) {

	assert(predictorEvaluationTask->GetClustering() != NULL);

	ost << endl << "Gravity centers :" << endl;
	ost << "Cluster";

	if (GetLearningExpertMode() and predictorEvaluationTask->GetClustering()->GetParameters()->GetWriteDetailedStatistics())
		ost << "\tInter L2\tInter L1\tInter cos.\tIntra L2\tIntra L1\tIntra cos.";

	ost << "\tFrequency\tCoverage\t";

	// affichage des modalités
	const ObjectArray& modalities = predictorEvaluationTask->GetClustering()->GetTargetAttributeValues();
	for (int i = 0; i < modalities.GetSize(); i++) {
		StringObject* o = cast(StringObject*, modalities.GetAt(i));
		ost << o->GetString() << "\t";
	}
	ost << endl;

	double totalInterL1 = 0.0;
	double totalInterL2 = 0.0;
	double totalInterCosinus = 0.0;
	double totalFrequency = 0.0;
	double totalCoverage = 0.0;
	ContinuousVector totalTargetValues;

	// trier les clusters sur les probas de la valeur cible, par ordre decroissant, mais sans toucher a l'ordre initial des clusters (sinon effets de bords possibles). On fait donc
	// une copie temporaire des clusters.
	// NB. le numero de cluster affiché après le tri, doit rester inchangé
	ObjectArray* clusters = new ObjectArray;
	for (int idxCluster = 0; idxCluster < predictorEvaluationTask->GetClustering()->GetClusters()->GetSize(); idxCluster++) {
		clusters->Add(predictorEvaluationTask->GetClustering()->GetClusters()->GetAt(idxCluster));
	}
	clusters->SetCompareFunction(KMCompareTargetProbs);
	clusters->Sort();

	// afficher les statistiques des clusters
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KMCluster* c = cast(KMCluster*, clusters->GetAt(idxCluster));

		ost << "Cluster " << c->GetLabel() << "\t";

		if (GetLearningExpertMode() and predictorEvaluationTask->GetClustering()->GetParameters()->GetWriteDetailedStatistics()) {

			double inertyInterL2 = (c->GetFrequency() == 0 ? 0 : c->GetInertyInter(KMParameters::L2Norm));
			totalInterL2 += inertyInterL2;
			ost << inertyInterL2 << "\t";

			double inertyInterL1 = (c->GetFrequency() == 0 ? 0 : c->GetInertyInter(KMParameters::L1Norm));
			totalInterL1 += inertyInterL1;
			ost << inertyInterL1 << "\t";

			double inertyInterCosinus = (c->GetFrequency() == 0 ? 0 : c->GetInertyInter(KMParameters::CosineNorm));
			totalInterCosinus += inertyInterCosinus;
			ost << inertyInterCosinus << "\t";

			double inertyIntraL2 = (c->GetFrequency() == 0 ? 0 : c->GetInertyIntra(KMParameters::L2Norm));
			ost << inertyIntraL2 << "\t";

			double inertyIntraL1 = (c->GetFrequency() == 0 ? 0 : c->GetInertyIntra(KMParameters::L1Norm));
			ost << inertyIntraL1 << "\t";

			double inertyIntraCosinus = (c->GetFrequency() == 0 ? 0 : c->GetInertyIntra(KMParameters::CosineNorm));
			ost << inertyIntraCosinus << "\t";
		}

		ost << c->GetFrequency() << "\t";
		totalFrequency += c->GetFrequency();

		ost << (c->GetFrequency() == 0 ? 0 : c->GetCoverage(lInstanceEvaluationNumber)) << "\t";
		totalCoverage += (c->GetFrequency() == 0 ? 0 : c->GetCoverage(lInstanceEvaluationNumber));

		// affichage des probas des modalités de la valeur cible
		// NB. ces probas sont issues de l'apprentissage OU du dico de modelisation, et non de l'evaluation elle même
		const ContinuousVector& targetValues = c->GetTargetProbs();
		totalTargetValues.SetSize(targetValues.GetSize());

		for (int i = 0; i < targetValues.GetSize(); i++) {
			ost << (c->GetFrequency() == 0 ? 0 : targetValues.GetAt(i)) << "\t";
			totalTargetValues.SetAt(i, totalTargetValues.GetAt(i) + ((c->GetFrequency() == 0 ? 0 : targetValues.GetAt(i)) * c->GetFrequency()));
		}

		ost << endl;
	}

	delete clusters;

	ost << "Total";

	if (GetLearningExpertMode() and predictorEvaluationTask->GetClustering()->GetParameters()->GetWriteDetailedStatistics()) {

		ost << "\t" << totalInterL2 << "\t"
			<< totalInterL1 << "\t"
			<< totalInterCosinus << "\t"
			<< predictorEvaluationTask->GetClustering()->GetClustersDistanceSum(KMParameters::L2Norm) / lInstanceEvaluationNumber << "\t"
			<< predictorEvaluationTask->GetClustering()->GetClustersDistanceSum(KMParameters::L1Norm) / lInstanceEvaluationNumber << "\t"
			<< predictorEvaluationTask->GetClustering()->GetClustersDistanceSum(KMParameters::CosineNorm) / lInstanceEvaluationNumber;
	}
	ost << "\t"
		<< totalFrequency << "\t"
		<< totalCoverage << "\t";

	for (int i = 0; i < totalTargetValues.GetSize(); i++) {
		ost << totalTargetValues.GetAt(i) / totalFrequency << "\t";
	}

	ost << endl;

	if (GetLearningExpertMode() and predictorEvaluationTask->GetClustering()->GetParameters()->GetWriteDetailedStatistics()) {

		ost << endl << "Inerty\tL1\tL2\tCos" << endl;

		ost << "Total\t"
			<< (1.0 / (double)lInstanceEvaluationNumber) * predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetDistanceSum(KMParameters::L1Norm) << "\t"
			<< (1.0 / (double)lInstanceEvaluationNumber) * predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetDistanceSum(KMParameters::L2Norm) << "\t"
			<< (1.0 / (double)lInstanceEvaluationNumber) * predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetDistanceSum(KMParameters::CosineNorm);
	}
}

void KMClassifierEvaluation::WriteJSONFullReportFields(JSONFile* fJSON,
	const ALString& sEvaluationLabel, ObjectArray* oaPredictorEvaluations)
{
	ObjectArray oaSortedPredictorEvaluations;

	require(oaPredictorEvaluations != NULL);
	require(CheckPredictorEvaluations(oaPredictorEvaluations));

	// Titre et caracteristiques de la base d'evaluation
	fJSON->WriteKeyString("reportType", "Evaluation");
	fJSON->WriteKeyString("evaluationType", sEvaluationLabel);

	// Description du probleme d'apprentissage
	fJSON->BeginKeyObject("summary");
	fJSON->WriteKeyString("dictionary", GetClass()->GetName());

	// Base de donnees
	fJSON->WriteKeyString("database", GetDatabaseName());
	fJSON->WriteKeyLongint("instances", GetEvaluationInstanceNumber());

	// Cas ou l'attribut cible n'est pas renseigne
	if (GetTargetAttributeType() == KWType::None)
	{
		fJSON->WriteKeyString("learningTask", "Unsupervised analysis");
	}
	// Autres cas
	else
	{
		// Cas ou l'attribut cible est continu
		if (GetTargetAttributeType() == KWType::Continuous)
			fJSON->WriteKeyString("learningTask", "Regression analysis");

		// Cas ou l'attribut cible est categoriel
		else if (GetTargetAttributeType() == KWType::Symbol)
			fJSON->WriteKeyString("learningTask", "Classification analysis");
	}

	// Informations eventuelles sur l'attribut cible
	if (GetTargetAttributeName() != "")
	{
		fJSON->WriteKeyString("targetVariable", GetTargetAttributeName());
		if (GetTargetAttributeType() == KWType::Symbol and GetMainTargetModalityIndex() != -1)
			fJSON->WriteKeyString("mainTargetValue", GetMainTargetModality().GetValue());
	}

	// Fin de description du probleme d'apprentissage
	fJSON->EndObject();

	// Calcul des identifiants des rapports bases sur leur rang
	ComputeRankIdentifiers(oaPredictorEvaluations);

	// Tableau synthetique des performances des predicteurs
	WriteJSONArrayReport(fJSON, "predictorsPerformance", oaPredictorEvaluations, true);

	// Tableau detaille des performances des predicteurs
	WriteJSONDictionaryReport(fJSON, "predictorsDetailedPerformance", oaPredictorEvaluations, false);

	// Rapport sur les courbes de performance
	SelectPerformanceCurvesReport(oaPredictorEvaluations, &oaSortedPredictorEvaluations);
	if (oaSortedPredictorEvaluations.GetSize() > 0)
		WriteJSONPerformanceCurveReportArray(fJSON, &oaSortedPredictorEvaluations);

	// donnees specifiques clustering
	WriteJSONKMeanStatistics(fJSON);
}

// ecriture du rapport JSON d evaluation (train ou test)
void KMClassifierEvaluation::WriteJSONKMeanStatistics(JSONFile* fJSON) {

	assert(predictorEvaluationTask->GetClustering() != NULL);
	assert(predictorEvaluationTask != NULL);
	const KMParameters* parameters = predictorEvaluationTask->GetClustering()->GetParameters();
	assert(parameters != NULL);
	assert(trainedPredictor != NULL);
	assert(trainedPredictor->GetPredictorClass() != NULL);

	const ContinuousVector& globalGravity = predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetEvaluationCentroidValues();
	if (globalGravity.GetSize() == 0)
		return;// peut arriver sur l'evaluation de test, si on a fait un mauvais parametrage "discard" dans l'IHM

	fJSON->BeginKeyObject("clustering");
	fJSON->WriteKeyLongint("evaluatedInstancesNumber", lInstanceEvaluationNumber);

	// calcul du ratio inertie inter/inertie totale
	const double totalInerty = (1.0 / (double)lInstanceEvaluationNumber) * predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetDistanceSum(parameters->GetDistanceType());
	double inertyInter = 0;
	for (int idxCluster = 0; idxCluster < predictorEvaluationTask->GetClustering()->GetClusters()->GetSize(); idxCluster++) {
		KMCluster* c = cast(KMCluster*, predictorEvaluationTask->GetClustering()->GetClusters()->GetAt(idxCluster));
		inertyInter += c->GetInertyInter(parameters->GetDistanceType());
	}

	fJSON->BeginKeyObject("clusteringStatistics");
	fJSON->WriteKeyString("clustering", "KMean");
	fJSON->WriteKeyContinuous("meanDistance", predictorEvaluationTask->GetClustering()->GetMeanDistance());
	fJSON->WriteKeyContinuous("inertyInterDividedByInertyTotal", inertyInter / totalInerty);
	fJSON->WriteKeyContinuous("daviesBouldinL2Norm", predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetDaviesBouldin());
	fJSON->WriteKeyContinuous("ariByClusters", predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetARIByClusters());
	fJSON->WriteKeyContinuous("predictiveClustering", predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetPredictiveClustering());
	fJSON->EndObject();

	WriteJSONClustersGravityCenters(fJSON);

	if (parameters->GetWriteDetailedStatistics()) {

		TaskProgression::BeginTask();
		TaskProgression::SetTitle("Detailed statistics");
		TaskProgression::DisplayLabel("Writing JSON detailed statistics...");
		TaskProgression::DisplayProgression(0);

		// regenerer les attributs CellIndex (qui sont systematiquement nettoyes apres chaque evaluation, train ou test)
		KMTrainedPredictor::AddCellIndexAttributes(trainedPredictor);

		// tri des attributs par ordre decroissant de level (si supervisé), ou par nom (si non supervisé)
		boolean bSortOnLevel = false;
		boolean bHasNativeCategoricalAttributes = false;

		ObjectArray* oaAttributesList = new ObjectArray;

		KWAttribute* attribute = trainedPredictor->GetPredictorClass()->GetHeadAttribute();
		while (attribute != NULL)
		{
			if (parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL) {

				oaAttributesList->Add(attribute);

				if (attribute->GetConstMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()) > 0)
					bSortOnLevel = true;

				if (attribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL)) {
					if (attribute->GetType() == KWType::Symbol and not attribute->GetConstMetaData()->IsKeyPresent("TargetVariable"))
						bHasNativeCategoricalAttributes = true;
				}
			}

			trainedPredictor->GetPredictorClass()->GetNextAttribute(attribute);
		}

		oaAttributesList->SetCompareFunction(bSortOnLevel ? KMCompareLevel : KMCompareAttributeName);
		oaAttributesList->Sort();

		KMPredictorEvaluation::WriteJSONContinuousMeanValues(fJSON, predictorEvaluationTask->GetClustering(), oaAttributesList);
		TaskProgression::DisplayProgression(5);
		KMPredictorEvaluation::WriteJSONContinuousMedianValues(fJSON, predictorEvaluationTask->GetClustering(), oaAttributesList, predictorEvaluationTask->GetReadInstancesForMedianComputation(), lInstanceEvaluationNumber);
		TaskProgression::DisplayProgression(10);
		KMPredictorEvaluation::WriteJSONNativeAttributesProbs(fJSON, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), oaAttributesList);
		TaskProgression::DisplayProgression(60);
		KMPredictorEvaluation::WriteJSONPercentagePerLineNativeAttributesProbs(fJSON, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), oaAttributesList);
		TaskProgression::DisplayProgression(100);
		TaskProgression::EndTask();

		delete oaAttributesList;

		// nettoyer les attributs CellIndex
		KMPredictorEvaluation::CleanPredictorClass(trainedPredictor->GetPredictorClass());
	}

	fJSON->EndObject();
}

void KMClassifierEvaluation::WriteJSONClustersGravityCenters(JSONFile* fJSON) {

	assert(predictorEvaluationTask->GetClustering() != NULL);

	fJSON->BeginKeyArray("gravityCenters");

	// trier les clusters sur les probas de la valeur cible, par ordre decroissant, mais sans toucher a l'ordre initial des clusters (sinon effets de bords possibles). On fait donc
	// une copie temporaire des clusters.
	// NB. le numero de cluster affiché après le tri, doit rester inchangé
	ObjectArray* clusters = new ObjectArray;
	for (int idxCluster = 0; idxCluster < predictorEvaluationTask->GetClustering()->GetClusters()->GetSize(); idxCluster++) {
		clusters->Add(predictorEvaluationTask->GetClustering()->GetClusters()->GetAt(idxCluster));
	}
	clusters->SetCompareFunction(KMCompareTargetProbs);
	clusters->Sort();

	// afficher les statistiques des clusters
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KMCluster* c = cast(KMCluster*, clusters->GetAt(idxCluster));

		fJSON->BeginObject();
		fJSON->WriteKeyString("cluster", ALString("cluster") + c->GetLabel());
		fJSON->WriteKeyLongint("frequency", c->GetFrequency());
		fJSON->WriteKeyContinuous("coverage", (c->GetFrequency() == 0 ? 0 : c->GetCoverage(lInstanceEvaluationNumber)));

		// affichage des probas des modalités de la valeur cible
		// NB. ces probas sont issues de l'apprentissage OU du dico de modelisation, et non de l'evaluation elle même
		const ContinuousVector& targetValues = c->GetTargetProbs();
		const ObjectArray& modalities = predictorEvaluationTask->GetClustering()->GetTargetAttributeValues();

		for (int i = 0; i < targetValues.GetSize(); i++) {
			StringObject* o = cast(StringObject*, modalities.GetAt(i));
			fJSON->WriteKeyContinuous(o->GetString(), (c->GetFrequency() == 0 ? 0 : targetValues.GetAt(i)));
		}
		fJSON->EndObject();
	}
	fJSON->EndArray();

	delete clusters;
}

void KMClassifierEvaluation::Evaluate(KWPredictor* predictor, KWDatabase* database)
{
	boolean bOk = true;
	KWLearningSpec currentLearningSpec;
	KWClassDomain* currentDomain;
	KWClassDomain* evaluationDomain;
	KWDatabase* evaluationDatabase;

	require(predictor != NULL);
	require(database != NULL);
	require(predictor->IsTrained());
	require(KWType::IsPredictorType(predictor->GetTargetAttributeType()));
	require(database->GetObjects()->GetSize() == 0);

	int nTargetIndex;
	KWTrainedClassifier* classifier;

	// Memorisation des modalites cibles de l'index de la modalite cible principale
	classifier = predictor->GetTrainedClassifier();
	svTrainedTargetModalities.SetSize(classifier->GetTargetValueNumber());
	for (nTargetIndex = 0; nTargetIndex < classifier->GetTargetValueNumber(); nTargetIndex++)
	{
		svTrainedTargetModalities.SetAt(nTargetIndex, classifier->GetTargetValueAt(nTargetIndex));
		if (classifier->GetTargetValueAt(nTargetIndex) == predictor->GetMainTargetModality())
			nPredictorMainTargetModalityIndex = nTargetIndex;
	}

	// Initialisation des criteres d'evaluation
	InitializeCriteria();

	// Memorisation du contexte d'evaluation
	sPredictorName = predictor->GetObjectLabel();
	evaluationDatabaseSpec.CopyFrom(database);
	SetLearningSpec(predictor->GetLearningSpec());

	// Personnalisation du dictionnaire de deploiement pour l'evaluation
	trainedPredictor = cast(KMTrainedClassifier*, predictor->GetTrainedClassifier());
	trainedPredictor->PrepareDeploymentClass(true, true);

	// Changement du LearningSpec courant a celui du predicteur
	currentLearningSpec.CopyFrom(predictor->GetLearningSpec());

	// Mise en place du domaine d'evaluation du predicteur et compilation
	currentDomain = KWClassDomain::GetCurrentDomain();
	evaluationDomain = trainedPredictor->GetPredictorDomain();
	if (evaluationDomain != currentDomain)
	{
		evaluationDomain->SetName("Evaluation");
		KWClassDomain::SetCurrentDomain(evaluationDomain);
	}
	evaluationDomain->Compile();

	// Clonage de la base d'evaluation, pour ne pas interagir avec les spec d'apprentissage en cours
	evaluationDatabase = database->Clone();
	evaluationDatabase->SetClassName(trainedPredictor->GetPredictorClass()->GetName());

	// Parametrage de la base et de la classe d'evaluation
	predictor->GetLearningSpec()->SetDatabase(evaluationDatabase);
	predictor->GetLearningSpec()->SetClass(trainedPredictor->GetPredictorClass());

	// Lancement de la tache d'evaluation sous-traitante
	predictorEvaluationTask = cast(KMClassifierEvaluationTask*, CreatePredictorEvaluationTask());
	bOk = predictorEvaluationTask->Evaluate(cast(KMPredictor*, predictor), evaluationDatabase, this);

	// Restitution de l'etat initial
	predictor->GetLearningSpec()->CopyFrom(&currentLearningSpec);
	if (evaluationDomain != currentDomain)
		KWClassDomain::SetCurrentDomain(currentDomain);
	trainedPredictor->PrepareDeploymentClass(true, false);

	// Reinitialisation en cas d'echec
	if (bOk)
		bIsStatsComputed = true;
	else
		Initialize();

	// Nettoyage
	delete evaluationDatabase;
}


KWPredictorEvaluationTask* KMClassifierEvaluation::CreatePredictorEvaluationTask()
{
	return new KMClassifierEvaluationTask;
}

void KMClassifierEvaluation::SetInstanceEvaluationNumber(const longint l) {
	lInstanceEvaluationNumber = l;
}

int
KMCompareTargetProbs(const void* elem1, const void* elem2)
{
	KMCluster* cluster1 = (KMCluster*) * (Object**)elem1;
	KMCluster* cluster2 = (KMCluster*) * (Object**)elem2;

	// Comparaison de 2 clusters sur la proba associee a la "main target value"

	const ContinuousVector& targetProbs1 = cluster1->GetTargetProbs();
	const ContinuousVector& targetProbs2 = cluster2->GetTargetProbs();

	// prendre en compte le cas de clusters devenus vides en evaluation, et qui peuvent ne pas avoir de probas
	if (targetProbs1.GetSize() == 0)
		return 1;

	if (targetProbs2.GetSize() == 0)
		return -1;

	return (targetProbs1.GetAt(0) > targetProbs2.GetAt(0) ? -1 : 1);
}






