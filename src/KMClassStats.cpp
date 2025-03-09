// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMClassStats.h"

KMClassStats::KMClassStats() {

	parameters = NULL;
	iClusteringVariablesNumber = 0;
}

void KMClassStats::SetKMParameters(KMParameters* p) {

	parameters = p;
}

void KMClassStats::SetClusteringVariablesNumber(int i) {

	iClusteringVariablesNumber = i;
}

int KMClassStats::GetClusteringVariablesNumber() const {
	return iClusteringVariablesNumber;
}


void KMClassStats::WriteReport(ostream& ost)
{
	ObjectArray oaSymbolAttributeStats;
	ObjectArray oaContinuousAttributeStats;
	KWAttributeStats* attributeStats;
	int i;
	int nType;
	int nAttributeNumber;
	int nTotalAttributeNumber;

	require(Check());
	require(IsStatsComputed());
	require(GetClass()->GetUsedAttributeNumber() ==
		GetClass()->GetLoadedAttributeNumber());

	// Titre
	ost << "Descriptive statistics" << "\n";
	ost << "\n\n";

	// Description du probleme d'apprentissage
	ost << "Problem description" << "\n";
	ost << "\n";
	ost << "Dictionary" << "\t" << GetClass()->GetName() << "\n";

	// Nombres d'attributs par type
	ost << "Variables" << "\n";
	nTotalAttributeNumber = 0;
	for (nType = 0; nType < KWType::None; nType++)
	{
		if (KWType::IsData(nType))
		{
			nAttributeNumber = GetUsedAttributeNumberForType(nType);
			nTotalAttributeNumber += nAttributeNumber;
			if (nAttributeNumber > 0)
				ost << "\t" << KWType::ToString(nType) << "\t" << nAttributeNumber << "\n";
		}
	}
	ost << "\t" << "Total" << "\t" << nTotalAttributeNumber << "\n";
	ost << "\n";

	// Base de donnees
	ost << "Database\t" << GetDatabase()->GetDatabaseName() << "\n";
	ost << "Instances" << "\t" << GetInstanceNumber() << "\n";

	// Type de tâche d'apprentissage effectué
	ost << "\nLearning task";

	// Cas ou l'attribut cible n'est pas renseigne
	if (GetTargetAttributeType() == KWType::None)
	{
		ost << "\tUnsupervised analysis" << "\n";
	}
	// Autres cas
	else
	{
		// Cas ou l'attribut cible est continu
		if (GetTargetAttributeType() == KWType::Continuous)
			ost << "\tRegression analysis" << "\n";

		// Cas ou l'attribut cible est categoriel
		else if (GetTargetAttributeType() == KWType::Symbol)
			ost << "\tClassification analysis" << "\n";
	}

	// Parametrage eventuel de l'apprentissage supervise
	if (GetTargetAttributeName() != "")
	{
		// Attribut cible
		// On demande un affichage complet (source et cible) pour forcer
		// l'utilisation explicite du libelle "Target"
		assert(GetTargetValueStats()->GetSourceAttributeNumber() == 0);
		ost << "\n";
		GetTargetValueStats()->WriteAttributeArrayLineReports(ost, true, true);

		// Statistiques descriptives
		if (GetTargetAttributeType() == KWType::Continuous or
			(GetTargetAttributeType() == KWType::Symbol and
				GetTargetDescriptiveStats()->GetValueNumber() > GetTargetValueLargeNumber(GetInstanceNumber())))
		{
			ost << "\n";
			GetTargetDescriptiveStats()->WriteReport(ost);
		}

		// Detail par valeur dans le cas symbol
		if (GetTargetAttributeType() == KWType::Symbol)
		{
			ost << "\n";
			GetTargetValueStats()->WriteAttributePartArrayLineReports(ost, true, true);
		}
	}

	// Arret si base vide
	if (GetInstanceNumber() == 0)
		return;

	// Statistiques sur les nombres de variables evaluees, natives, construites, informatives
	if (GetWriteOptionStats1D())
	{
		ost << "\n";
		ost << "Evaluated variables" << "\t" << GetEvaluatedAttributeNumber() << "\n";
		if (GetConstructedAttributeNumber() > 0)
		{
			ost << "Native variables" << "\t" << GetNativeAttributeNumber() << "\n";
			ost << "Constructed variables" << "\t" << GetConstructedAttributeNumber() << "\n";
		}
		if (GetTargetAttributeName() != "" and
			parameters->GetCategoricalPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed and
			parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed)
			ost << "Informative variables" << "\t" << GetInformativeAttributeNumber() << "\n";

		ost << "Clustering input variables" << "\t" << iClusteringVariablesNumber << "\n";
	}


	// Algorithme utilises : ecraser les libelles Khiops des pre-traitements, par les libelles des pretraitements specifiques Enneade
	if (GetWriteOptionStats1D())
	{
		ost << "\n";
		ost << GetPreprocessingSpec()->GetDiscretizerSpec()->GetClassLabel() << "\t";
		if (IsTargetGrouped())
			ost << "MODL" << "\n";
		else {
			if (!parameters)
				ost << GetPreprocessingSpec()->GetDiscretizerSpec()->GetMethodLabel(GetTargetAttributeType()) << "\n";
			else
				ost << parameters->GetContinuousPreprocessingTypeLabel(true) << "\n";
		}

		ost << GetPreprocessingSpec()->GetGrouperSpec()->GetClassLabel() << "\t";
		if (IsTargetGrouped())
			ost << "MODL" << "\n";
		else {
			if (!parameters)
				ost << GetPreprocessingSpec()->GetGrouperSpec()->GetMethodLabel(GetTargetAttributeType()) << "\n";
			else
				ost << parameters->GetCategoricalPreprocessingTypeLabel(true) << "\n";
		}
	}

	// Cout du model null
	if (GetWriteOptionStats1D() and GetTargetAttributeName() != "")
	{
		ost << "\nNull model\n";
		ost << "\tConstr. cost\t" << GetNullConstructionCost() << "\n";
		ost << "\tPrep. cost\t" << GetNullPreparationCost() << "\n";
		ost << "\tData cost\t" << GetNullDataCost() << "\n";
	}

	// Calcul des identifiants des rapports bases sur leur rang
	ComputeRankIdentifiers(&oaAttributeStats);
	ComputeRankIdentifiers(GetAttributePairStats());

	// On dispatche les statistiques univariee par type d'attribut
	for (i = 0; i < oaAttributeStats.GetSize(); i++)
	{
		attributeStats = cast(KWAttributeStats*, oaAttributeStats.GetAt(i));
		if (attributeStats->GetAttributeType() == KWType::Symbol)
			oaSymbolAttributeStats.Add(attributeStats);
		else if (attributeStats->GetAttributeType() == KWType::Continuous)
			oaContinuousAttributeStats.Add(attributeStats);
	}

	// Rapports synthetiques
	if (GetWriteOptionStats1D())
		WriteArrayLineReport(ost, "Categorical variables statistics", &oaSymbolAttributeStats);
	if (GetWriteOptionStats1D())
		WriteArrayLineReport(ost, "Numerical variables statistics", &oaContinuousAttributeStats);
	if (GetWriteOptionStats2D())
		WriteArrayLineReport(ost, "Variables pairs statistics", GetAttributePairStats());

	// Rapports detailles
	if (GetWriteOptionStats1D())
		WriteArrayReport(ost, "Variables detailed statistics", &oaAttributeStats);
	if (GetWriteOptionStats2D())
		WriteArrayReport(ost, "Variables pairs detailed statistics\n(Pairs with two jointly informative variables)", GetAttributePairStats());
}

void KMClassStats::WriteJSONFields(JSONFile* fJSON)
{
	int nType;
	int nAttributeNumber;

	require(GetWriteOptionStats1D() != GetWriteOptionStats2D());
	require(Check());
	require(IsStatsComputed());
	require(GetClass()->GetUsedAttributeNumber() ==
		GetClass()->GetLoadedAttributeNumber());

	// Type de rapport
	if (GetWriteOptionStats1D())
		fJSON->WriteKeyString("reportType", "Preparation");
	else if (GetWriteOptionStats2D())
		fJSON->WriteKeyString("reportType", "BivariatePreparation");

	// Description du probleme d'apprentissage
	fJSON->BeginKeyObject("summary");
	fJSON->WriteKeyString("dictionary", GetClass()->GetName());

	// Nombres d'attributs par type
	fJSON->BeginKeyObject("variables");
	fJSON->BeginKeyArray("types");
	for (nType = 0; nType < KWType::None; nType++)
	{
		if (KWType::IsData(nType))
		{
			nAttributeNumber = GetUsedAttributeNumberForType(nType);
			if (nAttributeNumber > 0)
				fJSON->WriteString(KWType::ToString(nType));
		}
	}
	fJSON->EndArray();
	fJSON->BeginKeyArray("numbers");
	for (nType = 0; nType < KWType::None; nType++)
	{
		if (KWType::IsData(nType))
		{
			nAttributeNumber = GetUsedAttributeNumberForType(nType);
			if (nAttributeNumber > 0)
				fJSON->WriteInt(nAttributeNumber);
		}
	}
	fJSON->EndArray();
	fJSON->EndObject();

	// Base de donnees
	fJSON->WriteKeyString("database", GetDatabase()->GetDatabaseName());
	fJSON->WriteKeyInt("instances", GetInstanceNumber());

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

	// Parametrage eventuel de l'apprentissage supervise
	if (GetTargetAttributeName() != "")
	{
		// Attribut cible
		// On demande un affichage complet (source et cible) pour forcer
		// l'utilisation explicite du libelle "Target"
		assert(GetTargetValueStats()->GetSourceAttributeNumber() == 0);
		fJSON->WriteKeyString("targetVariable", GetTargetAttributeName());

		// Modalite cible pirncipale
		if (GetTargetAttributeType() == KWType::Symbol and GetMainTargetModalityIndex() != -1)
			fJSON->WriteKeyString("mainTargetValue", GetMainTargetModality().GetValue());

		// Statistiques descriptives
		GetTargetDescriptiveStats()->WriteJSONKeyReport(fJSON, "targetDescriptiveStats");

		// Detail par valeur dans le cas symbol
		if (GetTargetAttributeType() == KWType::Symbol)
		{
			GetTargetValueStats()->WriteJSONKeyValueFrequencies(fJSON, "targetValues");
		}
	}

	// Arret si base vide
	if (GetInstanceNumber() == 0)
	{
		fJSON->EndObject();
		return;
	}

	// Statistiques sur les nombres de variables evaluees, natives, construites, informatives
	if (GetWriteOptionStats1D())
	{
		fJSON->WriteKeyInt("evaluatedVariables", GetEvaluatedAttributeNumber());
		if (GetConstructedAttributeNumber() > 0)
		{
			fJSON->WriteKeyInt("nativeVariables", GetNativeAttributeNumber());
			fJSON->WriteKeyInt("constructedVariables", GetConstructedAttributeNumber());
		}
		if (GetTargetAttributeName() != "")
			fJSON->WriteKeyInt("informativeVariables", GetInformativeAttributeNumber());
	}

	// Algorithme utilises
	if (GetWriteOptionStats1D())
	{
		// Discretisation
		if (IsTargetGrouped())
			fJSON->WriteKeyString("discretization", "MODL");
		else {
			if (!parameters)
				fJSON->WriteKeyString("discretization", GetPreprocessingSpec()->GetDiscretizerSpec()->GetMethodLabel(GetTargetAttributeType()));
			else
				fJSON->WriteKeyString("discretization", parameters->GetContinuousPreprocessingTypeLabel(true));
		}

		// Groupement de valeur
		if (IsTargetGrouped())
			fJSON->WriteKeyString("valueGrouping", "MODL");
		else {
			if (!parameters)
				fJSON->WriteKeyString("valueGrouping", GetPreprocessingSpec()->GetGrouperSpec()->GetMethodLabel(GetTargetAttributeType()));
			else
				fJSON->WriteKeyString("valueGrouping", parameters->GetCategoricalPreprocessingTypeLabel(true));
		}
	}

	// Cout du model null
	if (GetWriteOptionStats1D() and GetTargetAttributeName() != "")
	{
		fJSON->BeginKeyObject("nullModel");
		fJSON->WriteKeyContinuous("constructionCost", GetNullConstructionCost());
		fJSON->WriteKeyContinuous("preparationCost", GetNullPreparationCost());
		fJSON->WriteKeyContinuous("dataCost", GetNullDataCost());
		fJSON->EndObject();
	}

	// Fin de description du probleme d'apprentissage
	fJSON->EndObject();

	// Calcul des identifiants des rapports bases sur leur rang
	ComputeRankIdentifiers(&oaAttributeStats);
	ComputeRankIdentifiers(GetAttributePairStats());

	// Rapports synthetiques
	if (GetWriteOptionStats1D())
		WriteJSONArrayReport(fJSON, "variablesStatistics", &oaAttributeStats, true);
	if (GetWriteOptionStats2D())
		WriteJSONArrayReport(fJSON, "variablesPairsStatistics", GetAttributePairStats(), true);

	// Rapports detailles
	if (GetWriteOptionStats1D())
		WriteJSONDictionaryReport(fJSON, "variablesDetailedStatistics", &oaAttributeStats, false);
	if (GetWriteOptionStats2D())
		WriteJSONDictionaryReport(fJSON, "variablesPairsDetailedStatistics", GetAttributePairStats(), false);
}




