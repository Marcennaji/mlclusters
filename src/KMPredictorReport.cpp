// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMPredictorReport.h"
#include "KMClusteringQuality.h"
#include "KMPredictorKNN.h"

///////////////////////////////////////////////////////////////////////////////
// Classe KMPredictorReport

KMPredictorReport::KMPredictorReport()
{
	kmTrainedClustering = NULL;
	predictor = NULL;
}


KMPredictorReport::~KMPredictorReport()
{
}

void KMPredictorReport::SetTrainedClustering(KMClustering* r)
{
	kmTrainedClustering = r;
}

KMClustering* KMPredictorReport::GetTrainedClustering() const
{
	return kmTrainedClustering;
}

void KMPredictorReport::WriteReport(ostream& ost)
{
	// Appel de la methode ancetre
	KWPredictorReport::WriteReport(ost);

	if (kmTrainedClustering == NULL)
		return; // peut arriver si on n'a pas appris le predicteur kmeans, et qu'on a genere un modele classifieur majoritaire a la place

	ost << endl << "Sample number percentage: " << ALString(DoubleToString(kmTrainedClustering->GetUsedSampleNumberPercentage())) << " %" << endl;

	const KMParameters* parameters = kmTrainedClustering->GetParameters();

	ost << endl << "Clustering parameters:";
	parameters->Write(ost);

	ost << endl << "K output value : " + ALString(IntToString(kmTrainedClustering->GetClusters()->GetSize()));
	ost << endl << "Best clustering obtained: " << endl;

	if (GetTargetAttributeName() != "")
		ost << endl << "EVA is " << ALString(DoubleToString(kmTrainedClustering->GetClusteringQuality()->GetEVA())) << endl;

	ost << endl << "Mean distance is " << ALString(DoubleToString(kmTrainedClustering->GetMeanDistance())) << endl << endl;

	WriteDaviesBouldin(ost);

	WriteCentroids(ost);

	WriteInitialCentroids(ost);

	WriteCenterRealInstances(ost);

	WriteCenterRealNativeInstances(ost);

	if (GetTargetAttributeName() != "" and
		parameters->GetCategoricalPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed and
		parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed)

		WriteLevels(ost);

}
void KMPredictorReport::WriteJSONKMeanReport(JSONFile* fJSON)
{
	assert(kmTrainedClustering->GetClusters()->GetSize() > 0);

	fJSON->BeginKeyObject("clustering");
	fJSON->WriteKeyDouble("sampleNumberPercentage", kmTrainedClustering->GetUsedSampleNumberPercentage());

	const KMParameters* parameters = kmTrainedClustering->GetParameters();

	parameters->WriteJSON(fJSON);

	fJSON->WriteKeyInt("kOutputValue", kmTrainedClustering->GetClusters()->GetSize());

	fJSON->BeginKeyObject("bestClustering");

	if (GetTargetAttributeName() != "") {
		double d = (kmTrainedClustering->GetClusteringQuality()->GetEVA() < 0 ? 0 : kmTrainedClustering->GetClusteringQuality()->GetEVA());
		fJSON->WriteKeyContinuous("eva", d);
	}

	fJSON->WriteKeyContinuous("meanDistance", kmTrainedClustering->GetMeanDistance());

	fJSON->EndObject();

	WriteJSONDaviesBouldin(fJSON);

	WriteJSONCentroids(fJSON);

	WriteJSONInitialCentroids(fJSON);

	WriteJSONCenterRealInstances(fJSON);

	WriteJSONCenterRealNativeInstances(fJSON);

	if (GetTargetAttributeName() != "" and
		parameters->GetCategoricalPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed and
		parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed)

		WriteJSONLevels(fJSON);

	fJSON->EndObject();

}


void KMPredictorReport::WriteJSONFullReportFields(JSONFile* fJSON,
	ObjectArray* oaTrainReports)
{
	KWPredictorReport* firstReport;

	require(oaTrainReports != NULL);
	require(CheckTrainReports(oaTrainReports));

	// Acces a la premiere evaluation
	firstReport = NULL;
	if (oaTrainReports->GetSize() > 0)
		firstReport = cast(KWPredictorReport*, oaTrainReports->GetAt(0));

	// Titre et caracteristiques de la base d'apprentissage
	fJSON->WriteKeyString("reportType", "Modeling");

	// Description du probleme d'apprentissage
	fJSON->BeginKeyObject("summary");
	fJSON->WriteKeyString("dictionary", firstReport->GetClass()->GetName());

	// Base de donnees
	fJSON->WriteKeyString("database", firstReport->GetDatabase()->GetDatabaseName());

	// Cas ou l'attribut cible n'est pas renseigne
	if (firstReport->GetTargetAttributeType() == KWType::None)
	{
		fJSON->WriteKeyString("learningTask", "Unsupervised analysis");
	}
	// Autres cas
	else
	{
		// Cas ou l'attribut cible est continu
		if (firstReport->GetTargetAttributeType() == KWType::Continuous)
			fJSON->WriteKeyString("learningTask", "Regression analysis");

		// Cas ou l'attribut cible est categoriel
		else if (firstReport->GetTargetAttributeType() == KWType::Symbol)
			fJSON->WriteKeyString("learningTask", "Classification analysis");
	}

	// Informations eventuelles sur l'attribut cible
	if (firstReport->GetTargetAttributeName() != "")
	{
		fJSON->WriteKeyString("targetVariable", firstReport->GetTargetAttributeName());
		if (firstReport->GetTargetAttributeType() == KWType::Symbol and firstReport->GetMainTargetModalityIndex() != -1)
			fJSON->WriteKeyString("mainTargetValue", firstReport->GetMainTargetModality().GetValue());
	}

	// Fin de description du probleme d'apprentissage
	fJSON->EndObject();

	// Calcul des identifiants des rapports bases sur leur rang
	ComputeRankIdentifiers(oaTrainReports);

	// Tableau synthetique des performances des predicteurs
	WriteJSONArrayReport(fJSON, "trainedPredictors", oaTrainReports, true);

	// Tableau detaille des performances des predicteurs
	WriteJSONDictionaryReport(fJSON, "trainedPredictorsDetails", oaTrainReports, false);

	// donnees specifiques KMean
	if (kmTrainedClustering != NULL)
		WriteJSONKMeanReport(fJSON);
}

void KMPredictorReport::WriteDaviesBouldin(ostream& ost) {

	const KMParameters* parameters = kmTrainedClustering->GetParameters();

	ost << endl << endl << "Davies Bouldin indexes, by attribute (L2 norm): " << endl;
	ost << endl << "Var name\tRecoded name\tDavies-Bouldin" << endl;

	const KMCluster* globalCluster = kmTrainedClustering->GetGlobalCluster();

	assert(globalCluster != NULL);
	assert(globalCluster->GetModelingCentroidValues().GetSize() > 0);

	for (int attrIdx = 0; attrIdx < globalCluster->GetModelingCentroidValues().GetSize(); attrIdx++) {

		if (not parameters->GetKMeanAttributesLoadIndexes().GetAt(attrIdx).IsValid())
			continue;

		ALString recodedAttributeName = parameters->GetLoadedAttributeNameByRank(attrIdx);

		ost << parameters->GetNativeAttributeName(recodedAttributeName) << "\t";
		ost << parameters->GetLoadedAttributeNameByRank(attrIdx) << "\t";
		ost << kmTrainedClustering->GetClusteringQuality()->GetDaviesBouldinForAttribute(attrIdx) << endl;
	}
}

void KMPredictorReport::WriteCentroids(ostream& ost) {

	const KMParameters* parameters = kmTrainedClustering->GetParameters();

	ost << endl << endl << "Centroids : " << endl;
	ost << endl << "Var name\tRecoded name\t";

	for (int i = 0; i < kmTrainedClustering->GetClusters()->GetSize(); i++) {

		KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(i));
		ost << "cluster " << c->GetLabel() << "\t";
	}
	ost << "global " << endl;

	const KMCluster* globalCluster = kmTrainedClustering->GetGlobalCluster();

	assert(globalCluster != NULL);
	assert(globalCluster->GetModelingCentroidValues().GetSize() > 0);

	for (int attrIdx = 0; attrIdx < globalCluster->GetModelingCentroidValues().GetSize(); attrIdx++) {

		KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(attrIdx);
		if (not loadIndex.IsValid())
			continue;

		ALString recodedAttributeName = parameters->GetLoadedAttributeNameByRank(attrIdx);

		ost << parameters->GetNativeAttributeName(recodedAttributeName) << "\t";
		ost << parameters->GetLoadedAttributeNameByRank(attrIdx) << "\t";

		for (int j = 0; j < kmTrainedClustering->GetClusters()->GetSize(); j++) {

			KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(j));
			ost << c->GetModelingCentroidValues().GetAt(attrIdx) << "\t";
		}
		ost << globalCluster->GetModelingCentroidValues().GetAt(attrIdx) << endl;
	}
}

void KMPredictorReport::WriteInitialCentroids(ostream& ost) {

	const KMParameters* parameters = kmTrainedClustering->GetParameters();

	ost << endl << endl << "Initial centroids (before convergence) : " << endl;
	ost << endl << "Var name\tRecoded name\t";

	for (int i = 0; i < kmTrainedClustering->GetClusters()->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(i));
		ost << "cluster " << c->GetLabel() << "\t";
	}

	ost << endl;

	const KMCluster* globalCluster = kmTrainedClustering->GetGlobalCluster();

	assert(globalCluster != NULL);
	assert(globalCluster->GetModelingCentroidValues().GetSize() > 0);

	for (int attrIdx = 0; attrIdx < globalCluster->GetModelingCentroidValues().GetSize(); attrIdx++) {

		if (not parameters->GetKMeanAttributesLoadIndexes().GetAt(attrIdx).IsValid())
			continue;

		ALString recodedAttributeName = parameters->GetLoadedAttributeNameByRank(attrIdx);

		ost << parameters->GetNativeAttributeName(recodedAttributeName) << "\t";
		ost << parameters->GetLoadedAttributeNameByRank(attrIdx) << "\t";

		for (int j = 0; j < kmTrainedClustering->GetClusters()->GetSize(); j++) {

			KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(j));
			ost << c->GetInitialCentroidValues().GetAt(attrIdx) << "\t";
		}

		ost << endl;
	}
}


void KMPredictorReport::WriteCenterRealInstances(ostream& ost) {

	//  affichage des instances reelles des centres de clusters, tous attributs confondus

	const KMParameters* parameters = kmTrainedClustering->GetParameters();

	ost << endl << "Real instances (nearest to centroids) : " << endl;
	ost << "Var name\tRecoded name\t";

	for (int i = 0; i < kmTrainedClustering->GetClusters()->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(i));
		ost << "cluster " << c->GetLabel() << "\t";
	}

	ost << "global " << endl;

	const KMCluster* globalCluster = kmTrainedClustering->GetGlobalCluster();

	assert(globalCluster != NULL);
	assert(globalCluster->GetModelingCentroidValues().GetSize() > 0);

	for (int attrIdx = 0; attrIdx < globalCluster->GetModelingCentroidValues().GetSize(); attrIdx++) {

		if (not parameters->GetKMeanAttributesLoadIndexes().GetAt(attrIdx).IsValid())
			continue;

		const ALString recodedAttributeName = parameters->GetLoadedAttributeNameByRank(attrIdx);
		const ALString nativeAttributeName = parameters->GetNativeAttributeName(recodedAttributeName);

		ost << nativeAttributeName << "\t";
		ost << recodedAttributeName << "\t";

		// affichage des valeurs des clusters pour cet attribut
		for (int j = 0; j < kmTrainedClustering->GetClusters()->GetSize(); j++) {

			KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(j));

			const KMClusterInstanceAttribute* kmAttribute = c->GetInstanceNearestToCentroid()->FindAttribute(nativeAttributeName, recodedAttributeName);

			assert(kmAttribute != NULL);

			if (kmAttribute->type == KWType::Continuous)
				ost << kmAttribute->continuousValue << "\t";
			else
				ost << kmAttribute->symbolicValue << "\t";
		}

		// affichage valeur cluster global
		const KMClusterInstanceAttribute* kmAttribute = globalCluster->GetInstanceNearestToCentroid()->FindAttribute(nativeAttributeName, recodedAttributeName);

		assert(kmAttribute != NULL);

		if (kmAttribute->type == KWType::Continuous)
			ost << kmAttribute->continuousValue << "\t";
		else
			ost << kmAttribute->symbolicValue << "\t";

		ost << endl;

	}

}

void KMPredictorReport::WriteCenterRealNativeInstances(ostream& ost) {

	// affichage des instances reelles des centres de clusters, uniquement les attributs natifs

	ost << endl << "Real native instances (nearest to centroids) : " << endl;
	ost << "Var name\t";

	for (int i = 0; i < kmTrainedClustering->GetClusters()->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(i));
		ost << "cluster " << c->GetLabel() << "\t";
	}

	ost << "global " << endl;

	const KMCluster* globalCluster = kmTrainedClustering->GetGlobalCluster();

	assert(globalCluster != NULL);
	assert(globalCluster->GetModelingCentroidValues().GetSize() > 0);

	const KMClusterInstance* instanceNearestToCentroid = globalCluster->GetInstanceNearestToCentroid();

	require(instanceNearestToCentroid != NULL);

	for (int i = 0; i < instanceNearestToCentroid->GetLoadedAttributes().GetSize(); i++) {

		KMClusterInstanceAttribute* attribute = cast(KMClusterInstanceAttribute*,
			instanceNearestToCentroid->GetLoadedAttributes().GetAt(i));

		if (attribute->recodedName != "" and attribute->recodedName != attribute->nativeName)
			continue;

		if (attribute->nativeName.GetLength() >= 9 and attribute->nativeName.Left(9) == "CellIndex")
			// cas particulier. Les attributs CellIndex ne sont pas des attributs recodés, ni des attributs natifs à proprement parler (meme s'ils sont traites comme tels).
			// Ils servent à produire certaines statistiques.
			continue;

		ost << attribute->nativeName << "\t";

		// affichage des valeurs des clusters pour cet attribut
		for (int j = 0; j < kmTrainedClustering->GetClusters()->GetSize(); j++) {

			KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(j));

			const KMClusterInstanceAttribute* kmAttribute =
				c->GetInstanceNearestToCentroid()->FindAttribute(attribute->nativeName, attribute->recodedName);

			assert(kmAttribute != NULL);

			if (kmAttribute->type == KWType::Continuous)
				ost << kmAttribute->continuousValue << "\t";
			else
				ost << kmAttribute->symbolicValue << "\t";
		}

		// affichage valeur cluster global
		const KMClusterInstanceAttribute* kmAttribute =
			globalCluster->GetInstanceNearestToCentroid()->FindAttribute(attribute->nativeName, attribute->recodedName);

		assert(kmAttribute != NULL);

		if (kmAttribute->type == KWType::Continuous)
			ost << kmAttribute->continuousValue << "\t";
		else
			ost << kmAttribute->symbolicValue << "\t";

		ost << endl;

	}
}

void KMPredictorReport::WriteLevels(ostream& ost) {

	assert(predictor != NULL);

	// pour chaque variable native, affichage du level de pretraitement et du level de clustering

	ost << endl << "Preprocessing and clustering levels : " << endl;
	ost << "Var name\tPreprocessing level\tClustering level" << endl;

	ObjectArray* stats = predictor->GetClassStats()->GetAttributeStats();

	for (int i = 0; i < stats->GetSize(); i++) {

		KWAttributeStats* stat = cast(KWAttributeStats*, stats->GetAt(i));

		if (stat->GetLevel() > 0) {

			Symbol s = ((ALString&)(stat->GetAttributeName())).GetBuffer(stat->GetAttributeName().GetLength());
			Object* o = kmTrainedClustering->GetClusteringLevelsDictionary().Lookup(s.GetNumericKey());
			if (o == NULL) {
				// peut arriver si on a parametre un nombre max d'attributs evalues
				continue;
			}
			ost << stat->GetAttributeName() << "\t" << stat->GetLevel();
			ContinuousObject* level = cast(ContinuousObject*, o);
			ost << "\t" << level->GetContinuous() << endl;
		}
	}
}

void KMPredictorReport::WriteJSONDaviesBouldin(JSONFile* fJSON) {

	const KMParameters* parameters = kmTrainedClustering->GetParameters();

	fJSON->BeginKeyArray("daviesBouldinIndexesL2Norm");

	const KMCluster* globalCluster = kmTrainedClustering->GetGlobalCluster();

	assert(globalCluster != NULL);
	assert(globalCluster->GetModelingCentroidValues().GetSize() > 0);

	for (int attrIdx = 0; attrIdx < globalCluster->GetModelingCentroidValues().GetSize(); attrIdx++) {

		if (not parameters->GetKMeanAttributesLoadIndexes().GetAt(attrIdx).IsValid())
			continue;

		ALString recodedAttributeName = parameters->GetLoadedAttributeNameByRank(attrIdx);

		fJSON->BeginObject();
		fJSON->WriteKeyString("nativeName", parameters->GetNativeAttributeName(recodedAttributeName));
		fJSON->WriteKeyString("recodedName", parameters->GetLoadedAttributeNameByRank(attrIdx));
		fJSON->WriteKeyContinuous("daviesBouldin", kmTrainedClustering->GetClusteringQuality()->GetDaviesBouldinForAttribute(attrIdx));
		fJSON->EndObject();
	}
	fJSON->EndArray();
}

void KMPredictorReport::WriteJSONCentroids(JSONFile* fJSON) {

	const KMParameters* parameters = kmTrainedClustering->GetParameters();

	fJSON->BeginKeyArray("centroids");

	const KMCluster* globalCluster = kmTrainedClustering->GetGlobalCluster();

	assert(globalCluster != NULL);
	assert(globalCluster->GetModelingCentroidValues().GetSize() > 0);

	for (int attrIdx = 0; attrIdx < globalCluster->GetModelingCentroidValues().GetSize(); attrIdx++) {

		KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(attrIdx);
		if (not loadIndex.IsValid())
			continue;

		fJSON->BeginObject();

		ALString recodedAttributeName = parameters->GetLoadedAttributeNameByRank(attrIdx);
		fJSON->WriteKeyString("varName", parameters->GetNativeAttributeName(recodedAttributeName));
		fJSON->WriteKeyString("recodedName", parameters->GetLoadedAttributeNameByRank(attrIdx));

		for (int j = 0; j < kmTrainedClustering->GetClusters()->GetSize(); j++) {

			KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(j));
			fJSON->WriteKeyContinuous(ALString("cluster") + c->GetLabel(), c->GetModelingCentroidValues().GetAt(attrIdx));
		}
		fJSON->WriteKeyContinuous("global", globalCluster->GetModelingCentroidValues().GetAt(attrIdx));

		fJSON->EndObject();
	}
	fJSON->EndArray();
}

void KMPredictorReport::WriteJSONInitialCentroids(JSONFile* fJSON) {

	const KMParameters* parameters = kmTrainedClustering->GetParameters();

	fJSON->BeginKeyArray("initialCentroids");

	const KMCluster* globalCluster = kmTrainedClustering->GetGlobalCluster();

	assert(globalCluster != NULL);
	assert(globalCluster->GetModelingCentroidValues().GetSize() > 0);

	for (int attrIdx = 0; attrIdx < globalCluster->GetModelingCentroidValues().GetSize(); attrIdx++) {

		if (not parameters->GetKMeanAttributesLoadIndexes().GetAt(attrIdx).IsValid())
			continue;

		fJSON->BeginObject();

		ALString recodedAttributeName = parameters->GetLoadedAttributeNameByRank(attrIdx);
		fJSON->WriteKeyString("varName", parameters->GetNativeAttributeName(recodedAttributeName));
		fJSON->WriteKeyString("recodedName", parameters->GetLoadedAttributeNameByRank(attrIdx));

		for (int j = 0; j < kmTrainedClustering->GetClusters()->GetSize(); j++) {

			KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(j));
			fJSON->WriteKeyContinuous(ALString("cluster") + c->GetLabel(), c->GetInitialCentroidValues().GetAt(attrIdx));
		}

		fJSON->EndObject();
	}
	fJSON->EndArray();
}


void KMPredictorReport::WriteJSONCenterRealInstances(JSONFile* fJSON) {

	//  affichage des instances reelles des centres de clusters, tous attributs confondus

	const KMParameters* parameters = kmTrainedClustering->GetParameters();

	fJSON->BeginKeyArray("realInstancesNearestToCentroids");

	const KMCluster* globalCluster = kmTrainedClustering->GetGlobalCluster();

	assert(globalCluster != NULL);
	assert(globalCluster->GetModelingCentroidValues().GetSize() > 0);

	const KMClusterInstance* instanceNearestToCentroid = globalCluster->GetInstanceNearestToCentroid();

	require(instanceNearestToCentroid != NULL);

	for (int i = 0; i < instanceNearestToCentroid->GetLoadedAttributes().GetSize(); i++) {

		KMClusterInstanceAttribute* attribute = cast(KMClusterInstanceAttribute*, instanceNearestToCentroid->GetLoadedAttributes().GetAt(i));

		if (not parameters->IsKMAttributeName(attribute->recodedName))
			continue;

		fJSON->BeginObject();

		fJSON->WriteKeyString("varName", attribute->nativeName);
		fJSON->WriteKeyString("recodedName", attribute->recodedName);

		// affichage des valeurs des clusters pour cet attribut
		for (int j = 0; j < kmTrainedClustering->GetClusters()->GetSize(); j++) {

			KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(j));

			const KMClusterInstanceAttribute* kmAttribute = c->GetInstanceNearestToCentroid()->FindAttribute(attribute->nativeName, attribute->recodedName);

			assert(kmAttribute != NULL);

			if (kmAttribute->type == KWType::Continuous)
				fJSON->WriteKeyContinuous(ALString("cluster") + c->GetLabel(), kmAttribute->continuousValue);
			else
				fJSON->WriteKeyString(ALString("cluster") + c->GetLabel(), kmAttribute->symbolicValue.GetValue());
		}

		// affichage valeur cluster global
		const KMClusterInstanceAttribute* kmAttribute = globalCluster->GetInstanceNearestToCentroid()->FindAttribute(attribute->nativeName, attribute->recodedName);

		assert(kmAttribute != NULL);

		if (kmAttribute->type == KWType::Continuous)
			fJSON->WriteKeyContinuous("global", kmAttribute->continuousValue);
		else
			fJSON->WriteKeyString("global", kmAttribute->symbolicValue.GetValue());

		fJSON->EndObject();
	}
	fJSON->EndArray();
}

void KMPredictorReport::WriteJSONCenterRealNativeInstances(JSONFile* fJSON) {

	// affichage des instances reelles des centres de clusters, uniquement les attributs natifs

	fJSON->BeginKeyArray("realNativeInstancesNearestToCentroids");


	const KMCluster* globalCluster = kmTrainedClustering->GetGlobalCluster();

	assert(globalCluster != NULL);
	assert(globalCluster->GetModelingCentroidValues().GetSize() > 0);

	const KMClusterInstance* instanceNearestToCentroid = globalCluster->GetInstanceNearestToCentroid();

	require(instanceNearestToCentroid != NULL);

	for (int i = 0; i < instanceNearestToCentroid->GetLoadedAttributes().GetSize(); i++) {

		KMClusterInstanceAttribute* attribute = cast(KMClusterInstanceAttribute*,
			instanceNearestToCentroid->GetLoadedAttributes().GetAt(i));

		if (attribute->recodedName != "" and attribute->recodedName != attribute->nativeName)
			continue;

		if (attribute->nativeName.GetLength() >= 9 and attribute->nativeName.Left(9) == "CellIndex")
			// cas particulier. Les attributs CellIndex ne sont pas des attributs recodés, ni des attributs natifs à proprement parler (meme s'ils sont traites comme tels).
			// Ils servent à produire certaines statistiques.
			continue;

		fJSON->BeginObject();

		fJSON->WriteKeyString("varName", attribute->nativeName);

		// affichage des valeurs des clusters pour cet attribut
		for (int j = 0; j < kmTrainedClustering->GetClusters()->GetSize(); j++) {

			KMCluster* c = cast(KMCluster*, kmTrainedClustering->GetClusters()->GetAt(j));

			const KMClusterInstanceAttribute* kmAttribute =
				c->GetInstanceNearestToCentroid()->FindAttribute(attribute->nativeName, attribute->recodedName);

			assert(kmAttribute != NULL);

			if (kmAttribute->type == KWType::Continuous)
				fJSON->WriteKeyContinuous(ALString("cluster") + c->GetLabel(), kmAttribute->continuousValue);
			else
				fJSON->WriteKeyString(ALString("cluster") + c->GetLabel(), kmAttribute->symbolicValue.GetValue());
		}

		// affichage valeur cluster global
		const KMClusterInstanceAttribute* kmAttribute =
			globalCluster->GetInstanceNearestToCentroid()->FindAttribute(attribute->nativeName, attribute->recodedName);

		assert(kmAttribute != NULL);

		if (kmAttribute->type == KWType::Continuous)
			fJSON->WriteKeyContinuous("global", kmAttribute->continuousValue);
		else
			fJSON->WriteKeyString("global", kmAttribute->symbolicValue.GetValue());

		fJSON->EndObject();

	}
	fJSON->EndArray();
}

void KMPredictorReport::WriteJSONLevels(JSONFile* fJSON) {

	assert(predictor != NULL);

	// pour chaque variable native, affichage du level de pretraitement et du level de clustering

	fJSON->BeginKeyArray("clusteringLevels");

	ObjectArray* stats = predictor->GetClassStats()->GetAttributeStats();

	for (int i = 0; i < stats->GetSize(); i++) {

		KWAttributeStats* stat = cast(KWAttributeStats*, stats->GetAt(i));

		if (stat->GetLevel() > 0) {

			Symbol s = ((ALString&)(stat->GetAttributeName())).GetBuffer(stat->GetAttributeName().GetLength());
			Object* o = kmTrainedClustering->GetClusteringLevelsDictionary().Lookup(s.GetNumericKey());
			if (o == NULL) {
				// peut arriver si on a parametre un nombre max d'attributs evalues
				continue;
			}
			fJSON->BeginObject();
			fJSON->WriteKeyString("varName", stat->GetAttributeName());
			fJSON->WriteKeyContinuous("preprocessingLevel", stat->GetLevel());
			ContinuousObject* level = cast(ContinuousObject*, o);
			fJSON->WriteKeyContinuous("clusteringLevel", level->GetContinuous());
			fJSON->EndObject();
		}
	}
	fJSON->EndArray();
}

void KMPredictorReport::SetPredictor(const KMPredictor* p) {
	predictor = p;
}


