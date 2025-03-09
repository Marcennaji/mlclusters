// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMPredictorEvaluation.h"
#include "KMClusteringQuality.h"
#include "KMPredictorEvaluationTask.h"

#include <sstream>

///////////////////////////////////////////////////////////////////////////////
// Classe KMPredictorEvaluation


KMPredictorEvaluation::KMPredictorEvaluation() {

	trainedPredictor = NULL;
	predictorEvaluationTask = NULL;
	lInstanceEvaluationNumber = 0;
}

KMPredictorEvaluation::~KMPredictorEvaluation() {

	if (predictorEvaluationTask != NULL)
		delete predictorEvaluationTask;
}



void KMPredictorEvaluation::WriteFullReport(ostream& ost,
	const ALString& sEvaluationLabel,
	ObjectArray* oaPredictorEvaluations)
{

	ObjectArray oaSortedPredictorEvaluations;

	require(oaPredictorEvaluations != NULL);
	require(CheckPredictorEvaluations(oaPredictorEvaluations));
	assert(predictorEvaluationTask != NULL);
	assert(predictorEvaluationTask->GetClustering() != NULL);

	// Titre et caracteristiques de la base d'evaluation
	ost << sEvaluationLabel << " ";
	ost << "evaluation report" << "\n";
	ost << "\n";
	ost << "Dictionary" << "\t" << GetClass()->GetName() << "\n";
	if (GetTargetAttributeName() != "")
	{
		ost << "Target variable" << "\t" << KWType::ToString(GetTargetAttributeType()) << "\t" << GetTargetAttributeName() << "\n";
		if (GetMainTargetModalityIndex() >= 0)
			ost << "Main target value" << "\t" << GetMainTargetModality() << "\n";
	}
	ost << "Database\t" << GetDatabaseName() << "\n";
	ost << "Instances\t" << GetEvaluationInstanceNumber() << "\n";

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

	CleanPredictorClass(trainedPredictor->GetPredictorClass());

}

void KMPredictorEvaluation::WriteJSONFullReportFields(JSONFile* fJSON,
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



void KMPredictorEvaluation::CleanPredictorClass(KWClass* predictorClass) {

	// passer en unused les attributs natifs du dico de modelisation, devenus inutiles
	// supprimer egalement les CellIndex cr��s temporairement pour g�n�rer les tableaux de frequences de modalit�s

	ObjectArray oaCellIndexes;

	KWAttribute* attribute = predictorClass->GetHeadAttribute();
	while (attribute != NULL)
	{
		if (attribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL) or
			attribute->GetConstMetaData()->IsKeyPresent(KMParameters::KM_ATTRIBUTE_LABEL)) {
			attribute->SetUsed(false);
			attribute->SetLoaded(false);
		}
		else
			if (attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::CELL_INDEX_METADATA))
				oaCellIndexes.Add(attribute);

		predictorClass->GetNextAttribute(attribute);
	}

	for (int i = 0; i < oaCellIndexes.GetSize(); i++) {
		KWAttribute* a = cast(KWAttribute*, oaCellIndexes.GetAt(i));
		predictorClass->DeleteAttribute(a->GetName());
	}

	predictorClass->Compile();
}

// ecriture du rapport d evaluation (train ou test)
void KMPredictorEvaluation::WriteKMeanStatistics(ostream& ost) {

	assert(predictorEvaluationTask->GetClustering() != NULL);
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
		<< "Clustering\tMean distance\tInerty inter / total\tDavies-Bouldin (L2)" << endl
		<< "KMean\t"
		<< ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetMeanDistance())) << "\t"
		<< ALString(DoubleToString(inertyInter / totalInerty)) << "\t"
		<< ALString(DoubleToString(predictorEvaluationTask->GetClustering()->GetClusteringQuality()->GetDaviesBouldin())) << endl << endl;

	WriteClustersGravityCenters(ost);

	if (GetLearningExpertMode() and predictorEvaluationTask->GetClustering()->GetParameters()->GetWriteDetailedStatistics()) {
		WriteClustersDistancesUnnormalized(ost, predictorEvaluationTask->GetClustering());
		WriteClustersDistancesNormalized(ost, predictorEvaluationTask->GetClustering());
		WriteTrainTestCentroidsShifting(ost, predictorEvaluationTask->GetClustering());
	}

	if (parameters->GetWriteDetailedStatistics()) {

		TaskProgression::BeginTask();
		TaskProgression::SetTitle("Detailed statistics");
		TaskProgression::DisplayLabel("Writing detailed statistics...");
		TaskProgression::DisplayProgression(0);

		// tri des attributs par ordre decroissant de level (si supervis�), ou par nom (si non supervis�)
		boolean bSortOnLevel = false;
		boolean bHasNativeCategoricalAttributes = false;

		ObjectArray* oaAttributesList = new ObjectArray;

		KWAttribute* attribute = trainedPredictor->GetPredictorClass()->GetHeadAttribute();
		while (attribute != NULL)
		{
			if (parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL) {

				oaAttributesList->Add(attribute);

				if (attribute->GetMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()) > 0)
					bSortOnLevel = true;

				if (attribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL)) {
					if (attribute->GetType() == KWType::Symbol)
						bHasNativeCategoricalAttributes = true;
				}
			}

			trainedPredictor->GetPredictorClass()->GetNextAttribute(attribute);
		}

		oaAttributesList->SetCompareFunction(bSortOnLevel ? KMCompareLevel : KMCompareAttributeName);
		oaAttributesList->Sort();

		WriteContinuousMeanValues(ost, predictorEvaluationTask->GetClustering(), oaAttributesList);
		TaskProgression::DisplayProgression(5);
		WriteContinuousMedianValues(ost, predictorEvaluationTask->GetClustering(), oaAttributesList, predictorEvaluationTask->GetReadInstancesForMedianComputation(), lInstanceEvaluationNumber);
		TaskProgression::DisplayProgression(10);

		if (GetLearningExpertMode() and bHasNativeCategoricalAttributes) {
			WriteCategoricalModeValues(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetAtomicModalitiesFrequencyTables(), oaAttributesList, trainedPredictor->GetPredictorClass());
			TaskProgression::DisplayProgression(20);
			WritePercentagePerLineModeValues(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetAtomicModalitiesFrequencyTables(), oaAttributesList);
		}
		TaskProgression::DisplayProgression(50);
		WriteNativeAttributesProbs(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), oaAttributesList);
		TaskProgression::DisplayProgression(60);
		WritePercentagePerLineNativeAttributesProbs(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), oaAttributesList);
		TaskProgression::DisplayProgression(70);

		if (GetLearningExpertMode()) {
			WriteCumulativeNativeAttributesProbs(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), true, oaAttributesList); // cumulatif ascendant
			TaskProgression::DisplayProgression(80);
			WriteCumulativeNativeAttributesProbs(ost, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), false, oaAttributesList);// cumulatif descendant
			TaskProgression::DisplayProgression(90);
			WriteGlobalGravityCenters(ost, predictorEvaluationTask->GetClustering());
		}

		TaskProgression::DisplayProgression(100);
		TaskProgression::EndTask();

		delete oaAttributesList;
	}
}

KWPredictorEvaluationTask* KMPredictorEvaluation::CreatePredictorEvaluationTask()
{
	return new KMPredictorEvaluationTask;
}

void KMPredictorEvaluation::WriteClustersGravityCenters(ostream& ost) {

	assert(predictorEvaluationTask->GetClustering() != NULL);

	ost << endl << "Gravity centers :" << endl;
	ost << "Cluster";

	if (GetLearningExpertMode() and predictorEvaluationTask->GetClustering()->GetParameters()->GetWriteDetailedStatistics())
		ost << "\tInter L2\tInter L1\tInter cos.\tIntra L2\tIntra L1\tIntra cos.";

	ost << "\tFrequency\tCoverage" << endl;

	double totalInterL1 = 0.0;
	double totalInterL2 = 0.0;
	double totalInterCosinus = 0.0;
	double totalFrequency = 0.0;
	double totalCoverage = 0.0;

	// afficher les statistiques des clusters
	for (int idxCluster = 0; idxCluster < predictorEvaluationTask->GetClustering()->GetClusters()->GetSize(); idxCluster++) {

		KMCluster* c = cast(KMCluster*, predictorEvaluationTask->GetClustering()->GetClusters()->GetAt(idxCluster));

		ost << "Cluster " << c->GetLabel() << "\t";

		if (GetLearningExpertMode() and predictorEvaluationTask->GetClustering()->GetParameters()->GetWriteDetailedStatistics()) {

			double inertyInterL2 = c->GetInertyInter(KMParameters::L2Norm);
			totalInterL2 += inertyInterL2;
			ost << inertyInterL2 << "\t";

			double inertyInterL1 = c->GetInertyInter(KMParameters::L1Norm);
			totalInterL1 += inertyInterL1;
			ost << inertyInterL1 << "\t";

			double inertyInterCosinus = c->GetInertyInter(KMParameters::CosineNorm);
			totalInterCosinus += inertyInterCosinus;
			ost << inertyInterCosinus << "\t";

			double inertyIntraL2 = c->GetInertyIntra(KMParameters::L2Norm);
			ost << inertyIntraL2 << "\t";

			double inertyIntraL1 = c->GetInertyIntra(KMParameters::L1Norm);
			ost << inertyIntraL1 << "\t";

			double inertyIntraCosinus = c->GetInertyIntra(KMParameters::CosineNorm);
			ost << inertyIntraCosinus << "\t";
		}

		ost << c->GetFrequency() << "\t";
		totalFrequency += c->GetFrequency();

		ost << c->GetCoverage(lInstanceEvaluationNumber);
		totalCoverage += c->GetCoverage(lInstanceEvaluationNumber);

		ost << endl;
	}

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

	ost << endl;

	if (GetLearningExpertMode() and predictorEvaluationTask->GetClustering()->GetParameters()->GetWriteDetailedStatistics()) {

		ost << endl << "Inerty\tL1\tL2\tCos" << endl;

		ost << "Total\t"
			<< (1.0 / (double)lInstanceEvaluationNumber) * predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetDistanceSum(KMParameters::L1Norm) << "\t"
			<< (1.0 / (double)lInstanceEvaluationNumber) * predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetDistanceSum(KMParameters::L2Norm) << "\t"
			<< (1.0 / (double)lInstanceEvaluationNumber) * predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetDistanceSum(KMParameters::CosineNorm);
	}
}

void KMPredictorEvaluation::Evaluate(KWPredictor* predictor, KWDatabase* database)
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

	// Initialisation des criteres d'evaluation
	InitializeCriteria();

	// Memorisation du contexte d'evaluation
	sPredictorName = predictor->GetObjectLabel();
	evaluationDatabaseSpec.CopyFrom(database);
	SetLearningSpec(predictor->GetLearningSpec());

	// Acces au predicteur appris
	trainedPredictor = cast(KMTrainedPredictor*, predictor->GetTrainedPredictor());

	// Personnalisation du dictionnaire de deploiement pour l'evaluation
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
	// Lors de la execution de la methode Evaluate de la tache el ecrit les resultats
	// directement dans l'objet courant car elle a ete declaree en tant que "friend"
	predictorEvaluationTask = cast(KMPredictorEvaluationTask*, CreatePredictorEvaluationTask());
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


void KMPredictorEvaluation::WriteNativeAttributesProbs(ostream& ost, const KMClustering* clustering, const ObjectDictionary& groupedModalitiesFrequencyTables, const ObjectArray* oaAttributesList) {

	// ecriture du tableau des probas par intervalles/modalit�s et par cluster, pour chaque attribut natif

	if (clustering->GetParameters()->GetVerboseMode())
		Global::AddSimpleMessage("\tWriting native attributes probas...");

	const ObjectDictionary& partitions = clustering->GetAttributesPartitioningManager()->GetPartitions();

	if (partitions.GetCount() == 0)
		return;

	const KMParameters* parameters = clustering->GetParameters();

	bool firstLine = true;

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		assert(parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL);

		IntObject* ioIndex = cast(IntObject*, parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()));

		const int iIndex = ioIndex->GetInt();

		// retrouver le nom de l'attribut natif
		ALString nativeName = parameters->GetNativeAttributeName(parameters->GetLoadedAttributeNameByRank(iIndex));

		if (nativeName == "")
			nativeName = parameters->GetLoadedAttributeNameByRank(iIndex);

		// retrouver la liste des modalit�s/intervalles de cet attribut, s'il y en a une
		ObjectArray* oaModalities = cast(ObjectArray*, partitions.Lookup(attribute->GetName()));

		if (oaModalities == NULL)
			continue;

		// retrouver la table de contingences pour cet attribut et ces modalit�s, s'il y en a une
		KWFrequencyTable* table = cast(KWFrequencyTable*, groupedModalitiesFrequencyTables.Lookup(attribute->GetName()));

		if (table == NULL)
			continue;

		if (firstLine) { // ligne d'entete

			ost << endl << "Native attributes probas : " << endl;
			ost << "Var name\tModality/Interval\t";

			for (int j = 0; j < clustering->GetClusters()->GetSize(); j++) {
				KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(j));
				ost << "cluster " << cluster->GetLabel() << "\t";
			}

			ost << "global " << endl;
			firstLine = false;
		}

		// pour chaque modalit�/intervalle de l'attribut
		for (int idxModality = 0; idxModality < table->GetFrequencyVectorNumber(); idxModality++) {

			StringObject* modalityLabel = cast(StringObject*, oaModalities->GetAt(idxModality));

			ost << nativeName << "\t" << modalityLabel->GetString() << "\t";

			double globalProba = 0.0;

			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(idxModality));

			//pour chaque cluster, afficher le comptage des modalit�s/intervalles
			for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

				// convertir le comptage en proba
				KMCluster* cluster = clustering->GetCluster(idxCluster);
				assert(cluster != NULL);
				double proba = (cluster->GetFrequency() == 0 ? 0 : (double)fv->GetFrequencyVector()->GetAt(idxCluster) / (double)cluster->GetFrequency());
				ost << proba << "\t";
				assert(proba <= 1);

				globalProba += (double)fv->GetFrequencyVector()->GetAt(idxCluster);

			}
			ost << globalProba / clustering->GetGlobalCluster()->GetFrequency() << endl;
		}
	}
}

void KMPredictorEvaluation::WriteCumulativeNativeAttributesProbs(ostream& ost, const KMClustering* clustering, const ObjectDictionary& groupedModalitiesFrequencyTables,
	const bool ascending, const ObjectArray* oaAttributesList) {

	// ecriture du tableau cumulatif des probas par intervalles/modalit�s et par cluster, pour chaque attribut natif

	if (clustering->GetParameters()->GetVerboseMode()) {
		if (ascending)
			Global::AddSimpleMessage("\tWriting cumulative ascending native attributes probas...");
		else
			Global::AddSimpleMessage("\tWriting cumulative descending native attributes probas...");
	}

	const ObjectDictionary& partitions = clustering->GetAttributesPartitioningManager()->GetPartitions();

	if (partitions.GetCount() == 0)
		return;

	const KMParameters* parameters = clustering->GetParameters();

	bool firstLine = true;

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		assert(parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL);

		IntObject* ioIndex = cast(IntObject*, parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()));

		const int iIndex = ioIndex->GetInt();

		// retrouver le nom de l'attribut natif
		ALString nativeName = parameters->GetNativeAttributeName(parameters->GetLoadedAttributeNameByRank(iIndex));

		if (nativeName == "")
			nativeName = parameters->GetLoadedAttributeNameByRank(iIndex);

		// retrouver la liste des modalit�s/intervalles de cet attribut, s'il y en a une
		ObjectArray* oaModalities = cast(ObjectArray*, partitions.Lookup(attribute->GetName()));

		if (oaModalities == NULL)
			continue;

		// retrouver la table de contingences pour cet attribut et ces modalit�s, s'il y en a une
		KWFrequencyTable* table = cast(KWFrequencyTable*, groupedModalitiesFrequencyTables.Lookup(attribute->GetName()));

		if (table == NULL)
			continue;

		if (firstLine) { // ligne d'entete

			ost << endl << "Cumulative " << (ascending ? "ascending" : "descending") << " - Table \"native attributes probas\" :" << endl;
			ost << "Var name\tModality/Interval\t";

			for (int j = 0; j < clustering->GetClusters()->GetSize(); j++) {
				KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(j));
				ost << "cluster " << cluster->GetLabel() << "\t";
			}

			ost << "global " << endl;
			firstLine = false;
		}

		// initaliser le tableau des probas cumulatives pour cet attribut
		ContinuousVector cvCumulativeProbas;
		cvCumulativeProbas.SetSize(clustering->GetClusters()->GetSize() + 1);// un cluster de plus, pour le cluster global

		if (ascending)
			// init a zero
			cvCumulativeProbas.Initialize();
		else {
			// init a 1
			for (int j = 0; j < cvCumulativeProbas.GetSize(); j++)
				cvCumulativeProbas.SetAt(j, 1);
		}

		const int idxGlobalCluster = clustering->GetClusters()->GetSize();

		// pour chaque modalit�/intervalle de l'attribut
		for (int idxModality = 0; idxModality < table->GetFrequencyVectorNumber(); idxModality++) {

			StringObject* modalityLabel = cast(StringObject*, oaModalities->GetAt(idxModality));

			ost << nativeName << "\t" << modalityLabel->GetString() << "\t";

			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(idxModality));

			double globalProba = 0;

			//pour chaque cluster, afficher le comptage des modalit�s/intervalles
			for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

				// convertir le comptage en proba
				KMCluster* cluster = clustering->GetCluster(idxCluster);
				assert(cluster != NULL);
				double proba = (cluster->GetFrequency() == 0 ? 0 : (double)fv->GetFrequencyVector()->GetAt(idxCluster) / (double)cluster->GetFrequency());

				globalProba += (double)fv->GetFrequencyVector()->GetAt(idxCluster);

				if (ascending) {
					cvCumulativeProbas.SetAt(idxCluster, cvCumulativeProbas.GetAt(idxCluster) + proba);
					ost << std::fixed << cvCumulativeProbas.GetAt(idxCluster) << "\t";
				}
				else {
					ost << std::fixed << cvCumulativeProbas.GetAt(idxCluster) << "\t";
					cvCumulativeProbas.SetAt(idxCluster, cvCumulativeProbas.GetAt(idxCluster) - proba);
				}
			}
			// affichage ppour le cluster global
			globalProba = globalProba / clustering->GetGlobalCluster()->GetFrequency();

			if (ascending) {
				cvCumulativeProbas.SetAt(idxGlobalCluster, cvCumulativeProbas.GetAt(idxGlobalCluster) + globalProba);
				ost << std::fixed << cvCumulativeProbas.GetAt(idxGlobalCluster) << endl;
			}
			else {
				ost << std::fixed << cvCumulativeProbas.GetAt(idxGlobalCluster) << endl;
				cvCumulativeProbas.SetAt(idxGlobalCluster, cvCumulativeProbas.GetAt(idxGlobalCluster) - globalProba);
			}
		}
	}
}

void KMPredictorEvaluation::WriteGlobalGravityCenters(ostream& ost, const KMClustering* clustering) {

	assert(clustering != NULL);
	const KMParameters* parameters = clustering->GetParameters();
	assert(parameters != NULL);

	const ContinuousVector& globalGravity = clustering->GetGlobalCluster()->GetEvaluationCentroidValues();

	const int nbAttr = globalGravity.GetSize();

	assert(nbAttr > 0);

	ost << endl << "Global gravity center : " << endl;

	for (int i = 0; i < nbAttr; i++) {
		if (parameters->GetKMeanAttributesLoadIndexes().GetAt(i).IsValid())
			ost << parameters->GetLoadedAttributeNameByRank(i) << "\t" << globalGravity.GetAt(i) << endl;
	}
}


void KMPredictorEvaluation::WritePercentagePerLineNativeAttributesProbs(ostream& ost, const KMClustering* clustering, const ObjectDictionary& groupedModalitiesFrequencyTables,
	const ObjectArray* oaAttributesList) {

	// ecriture du tableau des % par intervalles/modalit�s et par cluster, pour chaque attribut natif

	if (clustering->GetParameters()->GetVerboseMode())
		Global::AddSimpleMessage("\tWriting native attributes probas : percentages per lines...");

	const ObjectDictionary& partitions = clustering->GetAttributesPartitioningManager()->GetPartitions();

	if (partitions.GetCount() == 0)
		return;

	const KMParameters* parameters = clustering->GetParameters();

	bool firstLine = true;

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		assert(parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL);

		IntObject* ioIndex = cast(IntObject*, parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()));

		const int iIndex = ioIndex->GetInt();

		// retrouver le nom de l'attribut natif
		ALString nativeName = parameters->GetNativeAttributeName(parameters->GetLoadedAttributeNameByRank(iIndex));

		if (nativeName == "")
			nativeName = parameters->GetLoadedAttributeNameByRank(iIndex);

		// retrouver la liste des modalit�s/intervalles de cet attribut, s'il y en a une
		ObjectArray* oaModalities = cast(ObjectArray*, partitions.Lookup(attribute->GetName()));

		if (oaModalities == NULL)
			continue;

		// retrouver la table de contingences pour cet attribut et ces modalit�s, s'il y en a une
		KWFrequencyTable* table = cast(KWFrequencyTable*, groupedModalitiesFrequencyTables.Lookup(attribute->GetName()));

		if (table == NULL)
			continue;

		if (firstLine) { // ligne d'entete

			ost << endl << "Percentage per line - Native attributes proba : " << endl;
			ost << "Var name\tModality/Interval\t";

			for (int j = 0; j < clustering->GetClusters()->GetSize(); j++) {
				KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(j));
				ost << "cluster " << cluster->GetLabel() << "\t";
			}

			ost << "global " << endl;
			firstLine = false;
		}

		// pour chaque modalit�/intervalle de l'attribut
		for (int idxModality = 0; idxModality < table->GetFrequencyVectorNumber(); idxModality++) {

			StringObject* modalityLabel = cast(StringObject*, oaModalities->GetAt(idxModality));

			ost << nativeName << "\t" << modalityLabel->GetString() << "\t";

			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(idxModality));

			// calculer l'effectif total ayant cette modalit�/intervalle
			int globalInstancesNumber = 0;
			for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++)
				globalInstancesNumber += fv->GetFrequencyVector()->GetAt(idxCluster);

			double globalProba = 0;
			//pour chaque cluster, afficher le % d'instances qui ont ces modalit�s/intervalles
			for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

				double proba = (globalInstancesNumber == 0 ? 0 : (double)fv->GetFrequencyVector()->GetAt(idxCluster) / (double)globalInstancesNumber);
				ost << proba << "\t";
				globalProba += (double)proba;
			}
			ost << globalProba << endl;
		}
	}

}


void KMPredictorEvaluation::WriteContinuousMeanValues(ostream& ost, const KMClustering* clustering, const ObjectArray* oaAttributesList) {

	// ecriture du tableau des moyennes de valeurs continues, par cluster, pour chaque attribut natif

	if (clustering->GetParameters()->GetVerboseMode())
		Global::AddSimpleMessage("\tWriting continuous mean values...");

	bool firstLine = true;

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		if (not attribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL) or attribute->GetType() != KWType::Continuous)
			continue;

		if (firstLine) { // ligne d'entete

			ost << endl << "Mean values for Numerical attributes : " << endl;
			ost << "Var name\t";

			for (int j = 0; j < clustering->GetClusters()->GetSize(); j++) {
				KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(j));
				ost << "cluster " << cluster->GetLabel() << "\t";
			}

			ost << "global\tMissing values" << endl;
			firstLine = false;
		}

		ost << attribute->GetName() << "\t";

		for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

			KMCluster* cluster = clustering->GetCluster(idxCluster);
			assert(cluster != NULL);

			ost << cluster->GetNativeAttributeContinuousMeanValue(attribute) << "\t";

		}
		ost << clustering->GetGlobalCluster()->GetNativeAttributeContinuousMeanValue(attribute) << "\t" << clustering->GetGlobalCluster()->GetMissingValues(attribute) << endl;
	}
}

void KMPredictorEvaluation::WriteContinuousMedianValues(ostream& ost, const KMClustering* clustering, const ObjectArray* oaAttributesList, const longint iReadInstancesForMedianComputation, const longint lInstanceEvaluationNumber) {

	// ecriture du tableau des medianes de valeurs continues, par cluster, pour chaque attribut natif

	if (iReadInstancesForMedianComputation == 0)
		return; // pas assez de memoire pour stocker les instances servant a calculer les medianes

	if (clustering->GetParameters()->GetVerboseMode())
		Global::AddSimpleMessage("\tWriting continuous median values...");

	bool firstLine = true;

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		if (not attribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL) or attribute->GetType() != KWType::Continuous)
			continue;

		if (firstLine) { // ligne d'entete

			ost << endl << "Median values for Numerical attributes ";
			if (iReadInstancesForMedianComputation < lInstanceEvaluationNumber)
				ost << "(approximation, based on " << iReadInstancesForMedianComputation << " instances) ";
			ost << ": " << endl;
			ost << "Var name\t";

			for (int j = 0; j < clustering->GetClusters()->GetSize(); j++) {
				KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(j));
				ost << "cluster " << cluster->GetLabel() << "\t";
			}

			ost << "global\tMissing values" << endl;
			firstLine = false;
		}

		ost << attribute->GetName() << "\t";

		for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

			KMCluster* cluster = clustering->GetCluster(idxCluster);
			assert(cluster != NULL);

			ost << cluster->GetNativeAttributeContinuousMedianValue(attribute) << "\t";

		}
		ost << clustering->GetGlobalCluster()->GetNativeAttributeContinuousMedianValue(attribute) << "\t" << clustering->GetGlobalCluster()->GetMissingValues(attribute) << endl;
	}

}

void KMPredictorEvaluation::WriteCategoricalModeValues(ostream& ost, const KMClustering* clustering,
	const ObjectDictionary& atomicModalitiesFrequencyTables, const ObjectArray* oaAttributesList, const KWClass* kwc) {

	// uniquement pour les variables categorielles qui ont au plus 10 modalit�s   :
	//   ecrire le % d'instances ayant cette valeur de modalit� (pour cette variable), sachant qu'ils sont dans le cluster n

	// ecriture du tableau des probas par intervalles/modalit�s et par cluster, pour chaque attribut natif


	if (clustering->GetParameters()->GetVerboseMode())
		Global::AddSimpleMessage("\tWriting categorical mode values...");

	const ObjectDictionary& atomicModalities = clustering->GetAttributesPartitioningManager()->GetAtomicModalities();

	if (atomicModalities.GetCount() == 0)
		return;

	const KMParameters* parameters = clustering->GetParameters();

	bool firstLine = true;

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		assert(parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL);

		IntObject* ioIndex = cast(IntObject*, parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()));

		const int iIndex = ioIndex->GetInt();

		// retrouver le nom de l'attribut natif
		ALString nativeName = parameters->GetNativeAttributeName(parameters->GetLoadedAttributeNameByRank(iIndex));

		if (nativeName == "")
			nativeName = parameters->GetLoadedAttributeNameByRank(iIndex);

		// retrouver la liste des modalit�s de cet attribut, s'il y en a une
		ObjectArray* oaModalities = cast(ObjectArray*, atomicModalities.Lookup(attribute->GetName()));

		if (oaModalities == NULL)
			continue;

		// retrouver la table de contingences pour cet attribut et ces modalit�s, s'il y en a une
		KWFrequencyTable* table = cast(KWFrequencyTable*, atomicModalitiesFrequencyTables.Lookup(attribute->GetName()));

		if (table == NULL)
			continue;

		if (firstLine) { // ligne d'entete

			ost << endl << "Mode values for Categorical attributes : " << endl;
			ost << "Var name\tModality\t";

			for (int j = 0; j < clustering->GetClusters()->GetSize(); j++) {
				KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(j));
				ost << "cluster " << cluster->GetLabel() << "\t";
			}

			ost << "global " << endl;
			firstLine = false;
		}

		// pour chaque modalit� de l'attribut
		for (int idxModality = 0; idxModality < table->GetFrequencyVectorNumber(); idxModality++) {

			StringObject* modalityLabel = cast(StringObject*, oaModalities->GetAt(idxModality));

			if (modalityLabel->GetString() == "") {
				// construire un libelle "Missing Value" en g�rant la presence eventuelle d'une modalit� ayant deja cette valeur.
				// NB: ne pas toucher a l'ordonnancement d'origine des modalites.
				ObjectArray* sortedModalities = oaModalities->Clone();
				sortedModalities->SetCompareFunction(KMCompareLabels);
				sortedModalities->Sort();
				StringObject* s = parameters->GetUniqueLabel(*sortedModalities, "Missing value");
				modalityLabel->SetString(s->GetString());
				delete s;
				delete sortedModalities;
			}

			ost << nativeName << "\t" << modalityLabel->GetString() << "\t";

			double globalProba = 0.0;

			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(idxModality));

			//pour chaque cluster, afficher le comptage des modalit�s/intervalles
			for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

				// convertir le comptage en proba
				KMCluster* cluster = clustering->GetCluster(idxCluster);
				assert(cluster != NULL);

				double proba = (cluster->GetFrequency() == 0 ? 0 : (double)fv->GetFrequencyVector()->GetAt(idxCluster) / (double)cluster->GetFrequency());

				ost << proba << "\t";
				assert(proba <= 1);

				globalProba += (double)fv->GetFrequencyVector()->GetAt(idxCluster);

			}
			ost << globalProba / clustering->GetGlobalCluster()->GetFrequency() << endl;
		}
	}

}

void KMPredictorEvaluation::WritePercentagePerLineModeValues(ostream& ost, const KMClustering* clustering, const ObjectDictionary& atomicModalitiesFrequencyTables, const ObjectArray* oaAttributesList) {

	// % d'instances �tant dans ce cluster, "sachant qu'ils ont cette modalit� pour cette variable"

	if (clustering->GetParameters()->GetVerboseMode())
		Global::AddSimpleMessage("\tWriting percentage per line mode values...");

	const ObjectDictionary& atomicModalities = clustering->GetAttributesPartitioningManager()->GetAtomicModalities();

	if (atomicModalities.GetCount() == 0)
		return;

	const KMParameters* parameters = clustering->GetParameters();

	bool firstLine = true;

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		assert(parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL);

		IntObject* ioIndex = cast(IntObject*, parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()));

		const int iIndex = ioIndex->GetInt();

		// retrouver le nom de l'attribut natif
		ALString nativeName = parameters->GetNativeAttributeName(parameters->GetLoadedAttributeNameByRank(iIndex));

		if (nativeName == "")
			nativeName = parameters->GetLoadedAttributeNameByRank(iIndex);

		// retrouver la liste des modalit�s de cet attribut, s'il y en a une
		ObjectArray* oaModalities = cast(ObjectArray*, atomicModalities.Lookup(attribute->GetName()));

		if (oaModalities == NULL)
			continue;

		// retrouver la table de contingences pour cet attribut et ces modalit�s, s'il y en a une
		KWFrequencyTable* table = cast(KWFrequencyTable*, atomicModalitiesFrequencyTables.Lookup(attribute->GetName()));

		if (table == NULL)
			continue;

		if (firstLine) { // ligne d'entete

			ost << endl << "Percentage Per Line - Mode values for Categorical attributes : " << endl;
			ost << "Var name\tModality\t";

			for (int j = 0; j < clustering->GetClusters()->GetSize(); j++) {
				KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(j));
				ost << "cluster " << cluster->GetLabel() << "\t";
			}

			ost << "global " << endl;
			firstLine = false;
		}

		// pour chaque modalit�/intervalle de l'attribut
		for (int idxModality = 0; idxModality < table->GetFrequencyVectorNumber(); idxModality++) {

			StringObject* modalityLabel = cast(StringObject*, oaModalities->GetAt(idxModality));

			ost << nativeName << "\t" << modalityLabel->GetString() << "\t";

			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(idxModality));

			// calculer l'effectif total ayant cette modalit�/intervalle
			int globalInstancesNumber = 0;
			for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++)
				globalInstancesNumber += fv->GetFrequencyVector()->GetAt(idxCluster);

			double globalProba = 0;
			//pour chaque cluster, afficher le % d'instances qui ont ces modalit�s/intervalles
			for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

				double proba = (globalInstancesNumber == 0 ? 0 : (double)fv->GetFrequencyVector()->GetAt(idxCluster) / (double)globalInstancesNumber);
				ost << proba << "\t";
				globalProba += (double)proba;
			}
			ost << globalProba << endl;
		}
	}
}

void KMPredictorEvaluation::WriteClustersDistancesUnnormalized(ostream& ost, KMClustering* clustering) {

	assert(clustering != NULL);

	const KMParameters* parameters = clustering->GetParameters();
	assert(parameters != NULL);

	ost << endl << endl << "Unnormalized distances between clusters centroids (" << parameters->GetDistanceTypeLabel() << ") :" << endl;

	Continuous** clustersCentersDistances = clustering->GetClustersCentersDistances();
	assert(clustersCentersDistances != NULL);

	const int nbClusters = clustering->GetClusters()->GetSize();

	// ecriture des colonnes
	for (int i = 0; i < nbClusters; i++) {
		KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(i));
		ost << "\t" << "cluster " << cluster->GetLabel();
	}

	ost << "\tglobal cluster" << endl;

	// ecriture des lignes
	for (int i = 0; i < nbClusters; i++) {

		KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(i));

		ost << "cluster " << cluster->GetLabel() << "\t";

		Continuous* line = clustersCentersDistances[i];

		for (int j = 0; j < nbClusters; j++) {

			ost << line[j] << "\t";
		}

		// distance de ce cluster avec le cluster global
		if (cluster->GetEvaluationCentroidValues().GetSize() == 0)
			// cluster devenu vide lors de l'evaluation
			ost << "?";
		else
			ost << clustering->GetDistanceBetween(clustering->GetGlobalCluster()->GetEvaluationCentroidValues(), cluster->GetEvaluationCentroidValues(), parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes());

		ost << endl;
	}

	// derniere ligne : cluster global
	ost << "global cluster" << "\t";
	for (int i = 0; i < nbClusters; i++) {

		KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(i));

		if (cluster->GetEvaluationCentroidValues().GetSize() == 0)
			// cluster devenu vide lors de l'evaluation
			ost << "0\t";
		else
			ost << clustering->GetDistanceBetween(clustering->GetGlobalCluster()->GetEvaluationCentroidValues(),
				cluster->GetEvaluationCentroidValues(), parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes()) << "\t";
	}
	ost << "0" << endl;

}


void KMPredictorEvaluation::WriteClustersDistancesNormalized(ostream& ost, KMClustering* clustering) {

	assert(clustering != NULL);

	const KMParameters* parameters = clustering->GetParameters();
	assert(parameters != NULL);

	ost << endl << endl << "Normalized distances between clusters centroids (" << parameters->GetDistanceTypeLabel() << ") :" << endl;

	Continuous** clustersCentersDistances = clustering->GetClustersCentersDistances();
	assert(clustersCentersDistances != NULL);

	const int nbClusters = clustering->GetClusters()->GetSize();

	// calcul de la distance la plus grande entre chaque cluster et le cluster global :

	Continuous maxDistanceBetweenGlobalCluster = 0;

	for (int i = 0; i < nbClusters; i++) {

		KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(i));

		if (cluster->GetEvaluationCentroidValues().GetSize() > 0) {
			const Continuous distance = clustering->GetDistanceBetween(clustering->GetGlobalCluster()->GetEvaluationCentroidValues(),
				cluster->GetEvaluationCentroidValues(), parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes());

			if (distance > maxDistanceBetweenGlobalCluster)
				maxDistanceBetweenGlobalCluster = distance;
		}
	}

	// ecriture des colonnes
	for (int i = 0; i < nbClusters; i++) {
		KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(i));
		ost << "\t" << "cluster " << cluster->GetLabel();
	}

	ost << "\tglobal cluster" << endl;

	// ecriture des lignes
	for (int i = 0; i < nbClusters; i++) {

		KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(i));

		ost << "cluster " << cluster->GetLabel() << "\t";

		Continuous* line = clustersCentersDistances[i];

		for (int j = 0; j < nbClusters; j++) {

			ost << (maxDistanceBetweenGlobalCluster > 0 ? line[j] / maxDistanceBetweenGlobalCluster : 0) << "\t";
		}

		// distance de ce cluster avec le cluster global
		if (cluster->GetEvaluationCentroidValues().GetSize() == 0)
			// cluster devenu vide lors de l'evaluation
			ost << "?";
		else {
			const Continuous distance = clustering->GetDistanceBetween(clustering->GetGlobalCluster()->GetEvaluationCentroidValues(),
				cluster->GetEvaluationCentroidValues(), parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes());
			ost << (maxDistanceBetweenGlobalCluster > 0 ? distance / maxDistanceBetweenGlobalCluster : 0);
		}

		ost << endl;
	}

	// derniere ligne : cluster global
	ost << "global cluster" << "\t";
	for (int i = 0; i < nbClusters; i++) {

		KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(i));

		if (cluster->GetEvaluationCentroidValues().GetSize() == 0)
			// cluster devenu vide lors de l'evaluation
			ost << "0\t";
		else {
			const Continuous distance = clustering->GetDistanceBetween(clustering->GetGlobalCluster()->GetEvaluationCentroidValues(),
				cluster->GetEvaluationCentroidValues(), parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes());
			ost << (maxDistanceBetweenGlobalCluster > 0 ? distance / maxDistanceBetweenGlobalCluster : 0) << "\t";
		}
	}
	ost << "0" << endl;

}

void KMPredictorEvaluation::WriteTrainTestCentroidsShifting(ostream& ost, KMClustering* clustering) {

	assert(clustering != NULL);

	const KMParameters* parameters = clustering->GetParameters();
	assert(parameters != NULL);

	ost << endl << endl << "Centroids shifting, between modeling and evaluation (" << parameters->GetDistanceTypeLabel() << ") :" << endl;

	const int nbClusters = clustering->GetClusters()->GetSize();

	// ecriture des colonnes
	ost << "\t";
	for (int i = 0; i < nbClusters; i++) {
		KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(i));
		ost << "cluster " << cluster->GetLabel() << "\t";
	}

	ost << "global cluster" << endl;

	// ecriture des valeurs non normalisees
	ost << "Unnormalized\t";
	for (int i = 0; i < nbClusters; i++) {

		KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(i));

		if (cluster->GetEvaluationCentroidValues().GetSize() == 0)
			// cluster devenu vide lors de l'evaluation
			ost << "0\t";
		else
			ost << clustering->GetDistanceBetween(cluster->GetModelingCentroidValues(),
				cluster->GetEvaluationCentroidValues(), parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes()) << "\t";
	}
	ost << clustering->GetDistanceBetween(clustering->GetGlobalCluster()->GetModelingCentroidValues(),
		clustering->GetGlobalCluster()->GetEvaluationCentroidValues(), parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes()) << endl;

	// ecriture des valeurs normalisees : memes valeurs que precedemment, mais divises par la distance entre le cluster du modele et le cluster global du modele
	ost << "Normalized\t";
	for (int i = 0; i < nbClusters; i++) {

		KMCluster* cluster = cast(KMCluster*, clustering->GetClusters()->GetAt(i));

		if (cluster->GetEvaluationCentroidValues().GetSize() == 0)
			// cluster devenu vide lors de l'evaluation
			ost << "0\t";
		else {
			const Continuous numerator = clustering->GetDistanceBetween(cluster->GetModelingCentroidValues(),
				cluster->GetEvaluationCentroidValues(), parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes());
			const Continuous denominator = clustering->GetDistanceBetween(cluster->GetModelingCentroidValues(),
				clustering->GetGlobalCluster()->GetModelingCentroidValues(), parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes());

			if (denominator == 0 or numerator == 0)
				ost << "0\t";
			else
				ost << numerator / denominator << "\t";
		}
	}
	ost << endl;
}

int KMPredictorEvaluation::ComputeReadPercentageForMedianComputation(const boolean bDetailedStatistics, const longint estimatedInstancesNumber, KWClass* kwc) {

	const longint dAvailableMemory = RMResourceManager::GetRemainingAvailableMemory();
	longint dAdditionalMemory = KMPredictor::ComputeRequiredMemory(estimatedInstancesNumber, kwc);

	if (bDetailedStatistics)
		dAdditionalMemory *= 2; // constat empirique

	if (dAdditionalMemory > dAvailableMemory) {

		// memoire insuffisante : on calcule les medianes sur un sous-ensemble des donnees
		const double newInstancesNumber = ((double)(dAvailableMemory / dAdditionalMemory)) * estimatedInstancesNumber;

		int readpercentage = (((double)newInstancesNumber / (double)estimatedInstancesNumber)) * 100;

		assert(readpercentage > 0 and readpercentage < 100);

		return readpercentage;
	}
	else
		return 100;
}

// ecriture du rapport d evaluation JSON (train ou test)
void KMPredictorEvaluation::WriteJSONKMeanStatistics(JSONFile* fJSON) {

	assert(predictorEvaluationTask->GetClustering() != NULL);
	assert(predictorEvaluationTask != NULL);
	const KMParameters* parameters = predictorEvaluationTask->GetClustering()->GetParameters();
	assert(parameters != NULL);
	assert(trainedPredictor != NULL);
	assert(trainedPredictor->GetPredictorClass() != NULL);

	const ContinuousVector& globalGravity = predictorEvaluationTask->GetClustering()->GetGlobalCluster()->GetEvaluationCentroidValues();
	if (globalGravity.GetSize() == 0)
		return;// peut arriver sur l'evaluation de test, si on a fait un mauvais parametrage "exclude sample" dans l'IHM

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
	fJSON->EndObject();

	WriteJSONClustersGravityCenters(fJSON);

	if (parameters->GetWriteDetailedStatistics()) {

		TaskProgression::BeginTask();
		TaskProgression::SetTitle("Detailed statistics");
		TaskProgression::DisplayLabel("Writing detailed statistics...");
		TaskProgression::DisplayProgression(0);

		// regenerer les attributs CellIndex (qui sont systematiquement nettoyes apres chaque evaluation, train ou test)
		KMTrainedPredictor::AddCellIndexAttributes(trainedPredictor);

		// tri des attributs par ordre decroissant de level (si supervis�), ou par nom (si non supervis�)
		boolean bSortOnLevel = false;
		boolean bHasNativeCategoricalAttributes = false;

		ObjectArray* oaAttributesList = new ObjectArray;

		KWAttribute* attribute = trainedPredictor->GetPredictorClass()->GetHeadAttribute();
		while (attribute != NULL)
		{
			if (parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL) {

				oaAttributesList->Add(attribute);

				if (attribute->GetMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()) > 0)
					bSortOnLevel = true;

				if (attribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL)) {
					if (attribute->GetType() == KWType::Symbol)
						bHasNativeCategoricalAttributes = true;
				}
			}

			trainedPredictor->GetPredictorClass()->GetNextAttribute(attribute);
		}

		oaAttributesList->SetCompareFunction(bSortOnLevel ? KMCompareLevel : KMCompareAttributeName);
		oaAttributesList->Sort();

		WriteJSONContinuousMeanValues(fJSON, predictorEvaluationTask->GetClustering(), oaAttributesList);
		TaskProgression::DisplayProgression(5);
		WriteJSONContinuousMedianValues(fJSON, predictorEvaluationTask->GetClustering(), oaAttributesList, predictorEvaluationTask->GetReadInstancesForMedianComputation(), lInstanceEvaluationNumber);
		TaskProgression::DisplayProgression(10);

		TaskProgression::DisplayProgression(50);
		WriteJSONNativeAttributesProbs(fJSON, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), oaAttributesList);
		TaskProgression::DisplayProgression(60);
		WriteJSONPercentagePerLineNativeAttributesProbs(fJSON, predictorEvaluationTask->GetClustering(), predictorEvaluationTask->GetGroupedModalitiesFrequencyTables(), oaAttributesList);
		TaskProgression::DisplayProgression(70);

		TaskProgression::DisplayProgression(100);
		TaskProgression::EndTask();

		delete oaAttributesList;

		// nettoyer les attributs CellIndex
		CleanPredictorClass(trainedPredictor->GetPredictorClass());
	}

	fJSON->EndObject();
}


void KMPredictorEvaluation::WriteJSONContinuousMeanValues(JSONFile* fJSON, const KMClustering* clustering, const ObjectArray* oaAttributesList) {

	// ecriture du tableau des moyennes de valeurs continues, par cluster, pour chaque attribut natif

	if (clustering->GetParameters()->GetVerboseMode())
		Global::AddSimpleMessage("\tWriting JSON continuous mean values...");

	fJSON->BeginKeyArray("continuousMeanValues");

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		if (not attribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL) or attribute->GetType() != KWType::Continuous)
			continue;

		fJSON->BeginObject();

		fJSON->WriteKeyString("varName", attribute->GetName());

		for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

			KMCluster* cluster = clustering->GetCluster(idxCluster);
			assert(cluster != NULL);

			fJSON->WriteKeyContinuous(ALString("cluster") + cluster->GetLabel(), cluster->GetNativeAttributeContinuousMeanValue(attribute));
		}
		fJSON->WriteKeyContinuous("global", clustering->GetGlobalCluster()->GetNativeAttributeContinuousMeanValue(attribute));
		fJSON->WriteKeyLongint("missingValues", clustering->GetGlobalCluster()->GetMissingValues(attribute));

		fJSON->EndObject();
	}
	fJSON->EndArray();
}

void KMPredictorEvaluation::WriteJSONContinuousMedianValues(JSONFile* fJSON, const KMClustering* clustering, const ObjectArray* oaAttributesList, const longint iReadInstancesForMedianComputation, const longint lInstanceEvaluationNumber) {

	// ecriture du tableau des medianes de valeurs continues, par cluster, pour chaque attribut natif

	if (iReadInstancesForMedianComputation == 0)
		return; // pas assez de memoire pour stocker les instances servant a calculer les medianes

	if (clustering->GetParameters()->GetVerboseMode())
		Global::AddSimpleMessage("\tWriting JSON continuous median values...");

	fJSON->BeginKeyArray("continuousMedianValues");

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		if (not attribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL) or attribute->GetType() != KWType::Continuous)
			continue;

		fJSON->BeginObject();

		fJSON->WriteKeyString("varName", attribute->GetName());

		for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

			KMCluster* cluster = clustering->GetCluster(idxCluster);
			assert(cluster != NULL);
			fJSON->WriteKeyContinuous(ALString("cluster") + cluster->GetLabel(), cluster->GetNativeAttributeContinuousMedianValue(attribute));
		}
		fJSON->WriteKeyContinuous("global", clustering->GetGlobalCluster()->GetNativeAttributeContinuousMedianValue(attribute));
		fJSON->WriteKeyLongint("missingValues", clustering->GetGlobalCluster()->GetMissingValues(attribute));

		fJSON->EndObject();
	}
	fJSON->EndArray();
}

void KMPredictorEvaluation::WriteJSONNativeAttributesProbs(JSONFile* fJSON, const KMClustering* clustering, const ObjectDictionary& groupedModalitiesFrequencyTables, const ObjectArray* oaAttributesList) {

	// ecriture du tableau des probas par intervalles/modalit�s et par cluster, pour chaque attribut natif

	const ObjectDictionary& partitions = clustering->GetAttributesPartitioningManager()->GetPartitions();

	if (partitions.GetCount() == 0)
		return;

	if (clustering->GetParameters()->GetVerboseMode())
		Global::AddSimpleMessage("\tWriting JSON native attributes probas...");

	fJSON->BeginKeyArray("nativeAttributesProbs");

	const KMParameters* parameters = clustering->GetParameters();

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		assert(parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL);

		IntObject* ioIndex = cast(IntObject*, parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()));

		const int iIndex = ioIndex->GetInt();

		// retrouver le nom de l'attribut natif
		ALString nativeName = parameters->GetNativeAttributeName(parameters->GetLoadedAttributeNameByRank(iIndex));

		if (nativeName == "")
			nativeName = parameters->GetLoadedAttributeNameByRank(iIndex);


		// retrouver la liste des modalit�s/intervalles de cet attribut, s'il y en a une
		ObjectArray* oaModalities = cast(ObjectArray*, partitions.Lookup(attribute->GetName()));

		if (oaModalities == NULL)
			continue;

		// retrouver la table de contingences pour cet attribut et ces modalit�s, s'il y en a une
		KWFrequencyTable* table = cast(KWFrequencyTable*, groupedModalitiesFrequencyTables.Lookup(attribute->GetName()));

		if (table == NULL)
			continue;

		// pour chaque modalit�/intervalle de l'attribut
		for (int idxModality = 0; idxModality < table->GetFrequencyVectorNumber(); idxModality++) {

			StringObject* modalityLabel = cast(StringObject*, oaModalities->GetAt(idxModality));

			fJSON->BeginObject();

			fJSON->WriteKeyString("varName", nativeName);
			fJSON->WriteKeyString("modalityOrInterval", modalityLabel->GetString());

			double globalProba = 0.0;

			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(idxModality));

			//pour chaque cluster, afficher le comptage des modalit�s/intervalles
			for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

				// convertir le comptage en proba
				KMCluster* cluster = clustering->GetCluster(idxCluster);
				assert(cluster != NULL);
				double proba = (cluster->GetFrequency() == 0 ? 0 : (double)fv->GetFrequencyVector()->GetAt(idxCluster) / (double)cluster->GetFrequency());
				assert(proba <= 1);
				fJSON->WriteKeyContinuous(ALString("cluster" + cluster->GetLabel()), proba);
				globalProba += (double)fv->GetFrequencyVector()->GetAt(idxCluster);

			}
			fJSON->WriteKeyContinuous("global", globalProba / clustering->GetGlobalCluster()->GetFrequency());
			fJSON->EndObject();
		}
	}
	fJSON->EndArray();
}

void KMPredictorEvaluation::WriteJSONPercentagePerLineNativeAttributesProbs(JSONFile* fJSON, const KMClustering* clustering, const ObjectDictionary& groupedModalitiesFrequencyTables,
	const ObjectArray* oaAttributesList) {

	// ecriture du tableau des % par intervalles/modalit�s et par cluster, pour chaque attribut natif

	if (clustering->GetParameters()->GetVerboseMode())
		Global::AddSimpleMessage("\tWriting JSON native attributes probas : percentages per lines...");

	const ObjectDictionary& partitions = clustering->GetAttributesPartitioningManager()->GetPartitions();

	if (partitions.GetCount() == 0)
		return;

	fJSON->BeginKeyArray("percentagePerLineNativeAttributesProbs");

	const KMParameters* parameters = clustering->GetParameters();

	for (int i = 0; i < oaAttributesList->GetSize(); i++)
	{
		KWAttribute* attribute = cast(KWAttribute*, oaAttributesList->GetAt(i));

		assert(parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()) != NULL);

		IntObject* ioIndex = cast(IntObject*, parameters->GetLoadedAttributesNames().Lookup(attribute->GetName()));

		const int iIndex = ioIndex->GetInt();

		// retrouver le nom de l'attribut natif
		ALString nativeName = parameters->GetNativeAttributeName(parameters->GetLoadedAttributeNameByRank(iIndex));

		if (nativeName == "")
			nativeName = parameters->GetLoadedAttributeNameByRank(iIndex);

		// retrouver la liste des modalit�s/intervalles de cet attribut, s'il y en a une
		ObjectArray* oaModalities = cast(ObjectArray*, partitions.Lookup(attribute->GetName()));

		if (oaModalities == NULL)
			continue;

		// retrouver la table de contingences pour cet attribut et ces modalit�s, s'il y en a une
		KWFrequencyTable* table = cast(KWFrequencyTable*, groupedModalitiesFrequencyTables.Lookup(attribute->GetName()));

		if (table == NULL)
			continue;

		// pour chaque modalit�/intervalle de l'attribut
		for (int idxModality = 0; idxModality < table->GetFrequencyVectorNumber(); idxModality++) {

			StringObject* modalityLabel = cast(StringObject*, oaModalities->GetAt(idxModality));

			fJSON->BeginObject();

			fJSON->WriteKeyString("varName", nativeName);
			fJSON->WriteKeyString("modalityOrInterval", modalityLabel->GetString());

			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(idxModality));

			// calculer l'effectif total ayant cette modalit�/intervalle
			int globalInstancesNumber = 0;
			for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++)
				globalInstancesNumber += fv->GetFrequencyVector()->GetAt(idxCluster);

			double globalProba = 0;
			//pour chaque cluster, afficher le % d'instances qui ont ces modalit�s/intervalles
			for (int idxCluster = 0; idxCluster < clustering->GetClusters()->GetSize(); idxCluster++) {

				double proba = (globalInstancesNumber == 0 ? 0 : (double)fv->GetFrequencyVector()->GetAt(idxCluster) / (double)globalInstancesNumber);
				KMCluster* cluster = clustering->GetCluster(idxCluster);
				assert(cluster != NULL);
				fJSON->WriteKeyContinuous(ALString("cluster" + cluster->GetLabel()), proba);

				globalProba += (double)proba;
			}
			fJSON->WriteKeyContinuous("global", globalProba);
			fJSON->EndObject();
		}
	}
	fJSON->EndArray();
}

void KMPredictorEvaluation::WriteJSONClustersGravityCenters(JSONFile* fJSON) {

	assert(predictorEvaluationTask->GetClustering() != NULL);

	fJSON->BeginKeyArray("gravityCenters");

	for (int idxCluster = 0; idxCluster < predictorEvaluationTask->GetClustering()->GetClusters()->GetSize(); idxCluster++) {

		KMCluster* c = cast(KMCluster*, predictorEvaluationTask->GetClustering()->GetClusters()->GetAt(idxCluster));

		fJSON->BeginObject();

		fJSON->WriteKeyString("cluster", ALString("cluster") + c->GetLabel());
		fJSON->WriteKeyInt("frequency", c->GetFrequency());
		fJSON->WriteKeyContinuous("coverage", c->GetCoverage(lInstanceEvaluationNumber));

		fJSON->EndObject();
	}

	fJSON->EndArray();
}

void KMPredictorEvaluation::SetInstanceEvaluationNumber(const longint l) {
	lInstanceEvaluationNumber = l;
}
























