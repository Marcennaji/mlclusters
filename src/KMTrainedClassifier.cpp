// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMTrainedClassifier.h"
#include "KMParametersView.h"

/////////////////////////////////////////////////////////////////////////////
// Classe KMTrainedClassifier

KMTrainedClassifier::KMTrainedClassifier() {

	kmModelingClustering = NULL;
	parameters = NULL;
}
KMTrainedClassifier::~KMTrainedClassifier() {

	if (kmModelingClustering != NULL)
		delete kmModelingClustering;

	if (parameters != NULL)
		delete parameters;
}

KMClustering* KMTrainedClassifier::CreateModelingClustering()
{
	// reconstruire un resultat K-Means et les parametres correspondants, à partir du dictionnaire de modelisation.

   // nettoyage avant (re)construction
	if (kmModelingClustering != NULL)
		delete kmModelingClustering;

	if (parameters != NULL)
		delete parameters;

	parameters = new KMParameters;
	kmModelingClustering = new KMClustering(parameters);

	parameters->SetIdClusterAttributeFromClass(predictorClass);
	if (parameters->GetIdClusterAttribute() == NULL)
		return NULL;// pas forcement une erreur, car si un kmeans n'a pas pu etre appris, alors on utilise un modele classifieur majoritaire

	if (parameters->GetIdClusterAttribute()->GetConstMetaData()->IsKeyPresent(KMParametersView::CATEGORICAL_PREPROCESSING_FIELD_NAME))
		parameters->SetCategoricalPreprocessingType(parameters->GetIdClusterAttribute()->GetConstMetaData()->GetStringValueAt(KMParametersView::CATEGORICAL_PREPROCESSING_FIELD_NAME));

	if (parameters->GetIdClusterAttribute()->GetConstMetaData()->IsKeyPresent(KMParametersView::CONTINUOUS_PREPROCESSING_FIELD_NAME))
		parameters->SetContinuousPreprocessingType(parameters->GetIdClusterAttribute()->GetConstMetaData()->GetStringValueAt(KMParametersView::CONTINUOUS_PREPROCESSING_FIELD_NAME));

	parameters->SetVerboseMode(parameters->GetIdClusterAttribute()->GetConstMetaData()->IsKeyPresent(KMParametersView::VERBOSE_MODE_FIELD_NAME));
	parameters->SetWriteDetailedStatistics(parameters->GetIdClusterAttribute()->GetConstMetaData()->IsKeyPresent(KMParametersView::DETAILED_STATISTICS_FIELD_NAME));

	if (parameters->GetWriteDetailedStatistics()) {
		// creer les attributs CellIndex, servant a produire les rapports de frequences de modalités
		if (parameters->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed and
			parameters->GetCategoricalPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed) {

			AddSimpleMessage("Attributes generation for detailed statistics (\"CellIndex\")");
			KMTrainedPredictor::AddCellIndexAttributes(this);
		}
	}

	// passer automatiquement en "used" et "loaded" les attributs supplementaires necessaires a l'evaluation, et memoriser
	// les index de chargement des attributs utilisés
	parameters->PrepareDeploymentClass(predictorClass);

	// creation du cluster "unique" des données, servant au calcul des stats globales
	kmModelingClustering->CreateGlobalCluster();

	// extraire les modalites/intervalles à partir du dico
	if (parameters->GetWriteDetailedStatistics())
		ExtractPartitions(predictorClass);

	if (KMTrainedPredictor::CreateClusters(predictorClass, kmModelingClustering) and CreateTargetValues())
		return kmModelingClustering;

	return NULL;

}

bool KMTrainedClassifier::CreateTargetValues() {

	assert(kmModelingClustering != NULL);
	assert(kmModelingClustering->GetGlobalCluster() != NULL);

	KWAttribute* attribute = predictorClass->GetHeadAttribute();
	while (attribute != NULL)
	{
		if (attribute->GetConstMetaData()->IsKeyPresent("Prediction")) {

			// retrouver l'attribut classifier correspondant :
			KWDerivationRuleOperand* operand = attribute->GetDerivationRule()->GetOperandAt(0);
			ALString classifierAttributeName = operand->GetAttributeName();
			KWAttribute* classifierAttribute = predictorClass->LookupAttribute(classifierAttributeName);
			assert(classifierAttribute != NULL);

			CreateTargetValuesAndTargetProbs(classifierAttribute);// si modele global, se baser sur le classifieur pour creer les valeurs cibles et les probas associees
			break;
		}

		// Attribut suivant
		predictorClass->GetNextAttribute(attribute);
	}

	return (kmModelingClustering->GetTargetAttributeValues().GetSize() == 0 ? false : true);
}

void KMTrainedClassifier::CreateTargetValuesAndTargetProbs(KWAttribute* classifierAttribute) {

	// extraire les repartitions des valeurs cibles issues de l'apprentissage
	// exemple de ligne pour un classifieur global : KMDRClassifier(IdCluster, ContinuousVector(0, 0, 1), ContinuousVector(0, 1, 0),..., SymbolValueSet("Iris-versicolor", "Iris-virginica", "Iris-setosa"))	;
	// exemple de ligne pour un classifieur de modeles locaux : LocalModelChooser(IdCluster, localModel_0_SNBClass, localModel_1_SNBClass, localModel_2_SNBClass,..., SymbolValueSet("Iris-versicolor", "Iris-virginica", "Iris-setosa"))	;

	for (int i = 1; i < classifierAttribute->GetDerivationRule()->GetOperandNumber(); i++) {

		KWDerivationRuleOperand* operand = classifierAttribute->GetDerivationRule()->GetOperandAt(i);

		if (operand->GetStructureName() == "Vector") {

			// si c'est un modle global (et non local), alors l'operande contient les probas apprises des valeurs cibles, pour le cluster de rang (i - 1)

			KWDRContinuousVector* cvRule = cast(KWDRContinuousVector*, operand->GetDerivationRule());
			ContinuousVector targetProbs; // probas associees aux valeurs cibles, pour un cluster donné

			for (int j = 0; j < cvRule->GetValues()->GetSize(); j++)
				targetProbs.Add(cvRule->GetValueAt(j));

			kmModelingClustering->GetCluster(i - 1)->SetTargetProbs(targetProbs);

		}
		else {

			if (operand->GetStructureName() == "ValueSetC") {

				// l'operande contient la liste des valeurs de l'attribut cible, rencontrees en apprentissage. Si l'apprentissage avait specifie une valeur cible,
				// alors cette valeur est la premiere de la liste

				KWDRSymbolValueSet* symbolRule = cast(KWDRSymbolValueSet*, operand->GetDerivationRule());
				ObjectArray targetValues; // valeurs de l'attribut cible

				for (int j = 0; j < symbolRule->GetValueNumber(); j++) {
					StringObject* value = new StringObject;
					value->SetString(symbolRule->GetValueAt(j).GetValue());
					targetValues.Add(value);
				}

				kmModelingClustering->SetTargetAttributeValues(targetValues);
			}
		}
	}
}


void KMTrainedClassifier::ExtractPartitions(KWClass* kwc) {

	// Parcours du dictionnaire de modelisation pour identifier les attributs necessaires
	KWAttribute* attribute = kwc->GetHeadAttribute();
	while (attribute != NULL)
	{
		if (attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::CELL_INDEX_METADATA) and attribute->GetLoaded() and attribute->GetUsed()) {

			assert(attribute->GetDerivationRule() != NULL and attribute->GetDerivationRule()->GetName() == "CellIndex");

			// analyser la regle de derivation pour savoir a quel type de pretraitement on a affaire
			KWDerivationRule* kwdr = attribute->GetDerivationRule();
			KWDerivationRuleOperand* operand = kwdr->GetSecondOperand();

			KWAttribute* nativeAttribute = kwc->LookupAttribute(operand->GetAttributeName());

			assert(nativeAttribute != NULL);

			if (nativeAttribute->GetType() == KWType::Continuous)
				ExtractSourceConditionalInfoContinuous(attribute, nativeAttribute);
			else
				if (nativeAttribute->GetType() == KWType::Symbol)
					ExtractSourceConditionalInfoCategorical(attribute, nativeAttribute);

		}
		kwc->GetNextAttribute(attribute);
	}
}

void KMTrainedClassifier::ExtractSourceConditionalInfoCategorical(const KWAttribute* attribute, const KWAttribute* nativeAttribute) {

	assert(attribute->GetLoaded() and attribute->GetUsed());

	// on a une ligne de dico comme : Continuous CellIndexVClass	 = CellIndex(VClass, Class)

	ALString originalAttributeName = attribute->GetDerivationRule()->GetFirstOperand()->GetAttributeName();
	KWAttribute* originalAttribute = predictorClass->LookupAttribute(originalAttributeName);
	assert(originalAttribute != NULL);

	// memoriser le nom d'attribut natif
	parameters = ((KMParameters*)(kmModelingClustering->GetParameters()));
	parameters->AddRecodedAttribute(nativeAttribute, attribute);
	parameters->AddRecodedAttribute(nativeAttribute, originalAttribute);
	parameters->SetCategoricalPreprocessingType(KMParameters::PreprocessingType::AutomaticallyComputed);// reconstituer le parametrage


	// ne pas prendre en compte les attributs a level nul
	if (nativeAttribute->GetConstMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()) == 0)
		return;

	// recuperer les groupes
	if (originalAttribute->GetDerivationRule()->GetFirstOperand()->GetDerivationRule()->GetName() == "ValueGroups") { // attention, la variable cible a une autre regle de derivation

		KWDRValueGroups* kwdrGroups = cast(KWDRValueGroups*,
			originalAttribute->GetDerivationRule()->GetFirstOperand()->GetDerivationRule());

		kmModelingClustering->GetAttributesPartitioningManager()->AddValueGroups(kwdrGroups, attribute->GetName(), 3, true);

	}
}

void KMTrainedClassifier::ExtractSourceConditionalInfoContinuous(const KWAttribute* attribute, const KWAttribute* nativeAttribute) {

	assert(attribute->GetLoaded() and attribute->GetUsed());

	// on a une ligne de type :
	// Continuous	CellIndexPSepalLength	 = CellIndex(PSepalLength, SepalLength)	;

	ALString originalAttributeName = attribute->GetDerivationRule()->GetFirstOperand()->GetAttributeName();
	KWAttribute* originalAttribute = predictorClass->LookupAttribute(originalAttributeName);
	assert(originalAttribute != NULL);

	// memoriser le nom d'attribut natif
	parameters = ((KMParameters*)(kmModelingClustering->GetParameters()));
	parameters->AddRecodedAttribute(nativeAttribute, attribute);
	parameters->AddRecodedAttribute(nativeAttribute, originalAttribute);

	// ne pas prendre en compte les attributs a level nul
	if (nativeAttribute->GetConstMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey()) == 0)
		return;

	// originalAttribute est de la forme : Structure(DataGrid)	PSepalLength = DataGrid(IntervalBounds(5.45, 6.15), SymbolValueSet("Iris-setosa", "Iris-versicolor", "Iris-virginica"), Frequencies(45, 5, 0, 6, 28, 16, 1, 10, 39))
	KWDRIntervalBounds* kwdrIntervalBounds = cast(KWDRIntervalBounds*,
		originalAttribute->GetDerivationRule()->GetFirstOperand()->GetDerivationRule());

	kmModelingClustering->GetAttributesPartitioningManager()->AddIntervalBounds(kwdrIntervalBounds, attribute->GetName());
}



