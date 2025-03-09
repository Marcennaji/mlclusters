// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMTrainedPredictor.h"
#include "KMParametersView.h"


/////////////////////////////////////////////////////////////////////////////
// Classe KMTrainedPredictor

KMTrainedPredictor::KMTrainedPredictor() {

	kmModelingClustering = NULL;
	parameters = NULL;
}
KMTrainedPredictor::~KMTrainedPredictor() {

	if (kmModelingClustering != NULL)
		delete kmModelingClustering;

	if (parameters != NULL)
		delete parameters;

}

int KMTrainedPredictor::GetTargetType() const
{
	return KWType::None;
}

KMClustering* KMTrainedPredictor::CreateModelingClustering()
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

	if (parameters->GetIdClusterAttribute() == NULL) {
		AddWarning(ALString("Invalid clustering modeling dictionary : it has no ") + KMPredictor::ID_CLUSTER_METADATA + " attribute");
		return NULL;
	}

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
			AddCellIndexAttributes(this);
		}
	}

	// passer automatiquement en "used" et "loaded" les attributs supplementaires necessaires a l'evaluation, et memoriser
	// les index de chargement des attributs utilisés
	parameters->PrepareDeploymentClass(predictorClass);

	// creation cluster "unique" des données, servant au calcul des stats globales
	kmModelingClustering->CreateGlobalCluster();

	// extraire les modalites/intervalles à partir du dico
	if (parameters->GetWriteDetailedStatistics())
		ExtractPartitions(predictorClass);

	if (CreateClusters(predictorClass, kmModelingClustering))
		return kmModelingClustering;

	AddWarning("Invalid clustering modeling dictionary : can't recreate clusters and/or target values");
	return NULL;
}


bool KMTrainedPredictor::CreateClusters(KWClass* predictorClass, KMClustering* clustering) {

	KMParameters* parameters = (KMParameters*)clustering->GetParameters();

	ContinuousVector globalCentroid;
	globalCentroid.SetSize(predictorClass->GetLoadedAttributeNumber());
	require(globalCentroid.GetSize() != 0);
	globalCentroid.Initialize();

	KWAttribute* attribute = predictorClass->GetHeadAttribute();
	while (attribute != NULL)
	{
		if (attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::DISTANCE_CLUSTER_LABEL)) {

			KMCluster* cluster = KMTrainedPredictor::CreateCluster(attribute, parameters, predictorClass);

			if (cluster == NULL) {
				return false;
			}
			clustering->GetClusters()->Add(cluster);

			cluster->SetLabel(attribute->GetConstMetaData()->GetStringValueAt(KMPredictor::CLUSTER_LABEL));
		}

		// au passage, renseigner le centroide du cluster global (issu du modele)
		if (attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::GLOBAL_GRAVITY_CENTER_LABEL)) {

			const Continuous attributeGlobalGravity = attribute->GetConstMetaData()->GetDoubleValueAt(KMPredictor::GLOBAL_GRAVITY_CENTER_LABEL);

			Object* o = parameters->GetKMAttributeNames().Lookup(attribute->GetName());
			assert(o != NULL);
			IntObject* ioLoadIndex = cast(IntObject*, o);

			globalCentroid.SetAt(ioLoadIndex->GetInt(), attributeGlobalGravity);
		}

		// Attribut suivant
		predictorClass->GetNextAttribute(attribute);
	}

	parameters->SetKValue(clustering->GetClusters()->GetSize());

	clustering->GetGlobalCluster()->SetModelingCentroidValues(globalCentroid);

	return (clustering->GetClusters()->GetSize() == 0 ? false : true);

}

KMCluster* KMTrainedPredictor::CreateCluster(KWAttribute* distanceClusterAttribute, KMParameters* parameters,
	KWClass* predictorClass) {

	assert(parameters != NULL);
	assert(distanceClusterAttribute != NULL);
	assert(not distanceClusterAttribute->GetConstMetaData()->IsKeyPresent(KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL));

	ALString distanceLabel = distanceClusterAttribute->GetConstMetaData()->GetStringValueAt(KMPredictor::DISTANCE_CLUSTER_LABEL);
	assert(distanceLabel == "L1" or distanceLabel == "L2" or distanceLabel == "CO");

	if (distanceLabel == "L1")
		parameters->SetDistanceType(KMParameters::L1Norm);
	else
		if (distanceLabel == "L2")
			parameters->SetDistanceType(KMParameters::L2Norm);
		else
			if (distanceLabel == "CO")
				parameters->SetDistanceType(KMParameters::CosineNorm);

	if (parameters->GetDistanceType() == KMParameters::L1Norm or
		parameters->GetDistanceType() == KMParameters::L2Norm)

		return CreateClusterL1L2Norm(distanceClusterAttribute, parameters, predictorClass);
	else
		return CreateClusterCosineNorm(distanceClusterAttribute, parameters, predictorClass);


}

KMCluster* KMTrainedPredictor::CreateClusterL1L2Norm(KWAttribute* distanceClusterAttribute, KMParameters* parameters,
	KWClass* predictorClass) {

	assert(parameters != NULL);
	assert(distanceClusterAttribute != NULL);

	assert(parameters->GetDistanceType() == KMParameters::L1Norm or parameters->GetDistanceType() == KMParameters::L2Norm);

	KMCluster* cluster = new KMCluster(parameters);

	ContinuousVector clusterCentroids;
	clusterCentroids.SetSize(predictorClass->GetLoadedAttributeNumber());
	require(clusterCentroids.GetSize() != 0);
	clusterCentroids.Initialize();

	// extraire les valeurs des noms d'attributs et des centroides du cluster :

	for (int i = 0; i < distanceClusterAttribute->GetDerivationRule()->GetOperandNumber(); i++) {

		Continuous centroid = 0.0;
		ALString attributeName;

		// Norme L1 ou L2
		// NB. en norme L1, le premier operande est un Abs, et en norme L2, un Product. C'est ici transparent.
		KWDerivationRuleOperand* firstOperand = distanceClusterAttribute->GetDerivationRule()->GetOperandAt(i);
		KWDerivationRule* firstOperandRule = firstOperand->GetDerivationRule();
		KWDerivationRuleOperand* operandProduct = firstOperandRule->GetFirstOperand();
		KWDerivationRule* substractRule = operandProduct->GetDerivationRule();
		KWDerivationRuleOperand* operandsubstract = substractRule->GetFirstOperand();
		attributeName = operandsubstract->GetAttributeName();
		centroid = substractRule->GetSecondOperand()->GetContinuousConstant();

		KWAttribute* centroidAttribute = predictorClass->LookupAttribute(attributeName);
		if (centroidAttribute == NULL or not centroidAttribute->GetLoaded() or not centroidAttribute->GetUsed())
			return NULL;

		// ajouter ce centroide dans la case correspondant a l' index de chargement de l'attribut

		assert(parameters->IsKMeanAttributeLoadIndex(centroidAttribute->GetLoadIndex()));

		const int rank = parameters->GetAttributeRankFromLoadIndex(centroidAttribute->GetLoadIndex());
		clusterCentroids.SetAt(rank, centroid);
	}

	cluster->SetModelingCentroidValues(clusterCentroids);

	return cluster;
}

KMCluster* KMTrainedPredictor::CreateClusterCosineNorm(KWAttribute* distanceClusterAttribute, KMParameters* parameters,
	KWClass* predictorClass) {

	KMCluster* cluster = new KMCluster(parameters);

	ContinuousVector clusterCentroids;
	clusterCentroids.SetSize(predictorClass->GetLoadedAttributeNumber());
	require(clusterCentroids.GetSize() != 0);
	clusterCentroids.Initialize();

	// extraire les valeurs des noms d'attributs et des centroides du cluster :

	KWDerivationRuleOperand* sumOperand = distanceClusterAttribute->GetDerivationRule()->GetOperandAt(1)->GetDerivationRule()->GetOperandAt(0);
	assert(sumOperand != NULL);

	KWDerivationRule* sumOperandRule = sumOperand->GetDerivationRule();
	assert(sumOperandRule != NULL);

	for (int i = 0; i < sumOperandRule->GetOperandNumber(); i++) {

		Continuous centroid = 0.0;
		ALString attributeName;

		KWDerivationRuleOperand* operandProduct = sumOperandRule->GetOperandAt(i);
		KWDerivationRule* productRule = operandProduct->GetDerivationRule();
		attributeName = productRule->GetFirstOperand()->GetAttributeName();
		centroid = productRule->GetSecondOperand()->GetContinuousConstant();

		KWAttribute* centroidAttribute = predictorClass->LookupAttribute(attributeName);
		if (centroidAttribute == NULL or not centroidAttribute->GetLoaded() or not centroidAttribute->GetUsed())
			return NULL;

		const int rank = parameters->GetAttributeRankFromLoadIndex(centroidAttribute->GetLoadIndex());

		// ajouter ce centroide dans la case correspondant a l' index de chargement de l'attribut
		assert(parameters->IsKMeanAttributeLoadIndex(centroidAttribute->GetLoadIndex()) and
			clusterCentroids.GetAt(rank) == 0);

		clusterCentroids.SetAt(rank, centroid);
	}

	cluster->SetModelingCentroidValues(clusterCentroids);

	return cluster;
}


void KMTrainedPredictor::ExtractPartitions(KWClass* kwc) {

	// Parcours du dictionnaire de modelisation pour identifier les attributs necessaires, par leur libelle
	KWAttribute* attribute = kwc->GetHeadAttribute();
	while (attribute != NULL)
	{
		if (attribute->GetName().GetLength() >= 3 and attribute->GetName().Left(3) == "NRP")
			ExtractRankNormalization(attribute);

		if (attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::CELL_INDEX_METADATA)) { // attribut temporaire, créé uniquement pour produire des stats detaillees sur les variables categorielles
			assert(attribute->GetDerivationRule() != NULL and attribute->GetDerivationRule()->GetName() == "CellIndex");
			ExtractBasicGrouping(attribute);
		}

		// Attribut suivant
		kwc->GetNextAttribute(attribute);
	}
}

void KMTrainedPredictor::ExtractRankNormalization(const KWAttribute* attribute) {

	assert(attribute->GetLoaded() and attribute->GetUsed());

	// memoriser le nom d'attribut natif
	ALString nativeName = attribute->GetDerivationRule()->GetSecondOperand()->GetAttributeName();
	KWAttribute* nativeAttribute = predictorClass->LookupAttribute(nativeName);
	((KMParameters*)(kmModelingClustering->GetParameters()))->AddRecodedAttribute(nativeAttribute, attribute);

	KWDRIntervalBounds* kwdrIntervalBounds = cast(KWDRIntervalBounds*,
		attribute->GetDerivationRule()->GetFirstOperand()->GetDerivationRule()->GetFirstOperand()->GetDerivationRule());

	kmModelingClustering->GetAttributesPartitioningManager()->AddIntervalBounds(kwdrIntervalBounds, attribute->GetName());
}

void KMTrainedPredictor::ExtractBasicGrouping(const KWAttribute* attribute) {

	assert(attribute->GetLoaded() and attribute->GetUsed());

	// on a une ligne de dico comme : Continuous	IndexPworkclass	 = CellIndex(Pworkclass, workclass)
	// il faut "remonter" a l'attribut originel (ici, Pworkclass) , pour en extraire les valeurs de groupes
	ALString originalAttributeName = attribute->GetDerivationRule()->GetFirstOperand()->GetAttributeName();
	KWAttribute* originalAttribute = predictorClass->LookupAttribute(originalAttributeName);
	assert(originalAttribute != NULL);

	// memoriser le nom d'attribut natif
	ALString nativeName = attribute->GetDerivationRule()->GetSecondOperand()->GetAttributeName();
	KWAttribute* nativeAttribute = predictorClass->LookupAttribute(nativeName);
	parameters = ((KMParameters*)(kmModelingClustering->GetParameters()));
	parameters->AddRecodedAttribute(nativeAttribute, attribute);
	parameters->AddRecodedAttribute(nativeAttribute, originalAttribute);
	parameters->SetCategoricalPreprocessingType(KMParameters::PreprocessingType::AutomaticallyComputed);

	// recuperer les groupes, aux fins d'affichage dans les rapports
	// originalAttribute est de la forme :
	//		Structure(DataGrid)	Pworkclass	 = DataGrid(ValueGroups(ValueGroup("Self-emp-not-inc", "Local-gov", "Federal-gov", "State-gov", "Self-emp-inc", " * "), ValueGroup("Private")), SymbolValueSet("less", "more"), Frequencies(31, 122, 23, 24))	;
	KWDRValueGroups* kwdrGroups = cast(KWDRValueGroups*,
		originalAttribute->GetDerivationRule()->GetFirstOperand()->GetDerivationRule());

	kmModelingClustering->GetAttributesPartitioningManager()->AddValueGroups(kwdrGroups, attribute->GetName(), 3, false);

}

void KMTrainedPredictor::AddCellIndexAttributes(KWTrainedPredictor* trainedPredictor) {

	// pour chaque attribut préparé de type datagrid, ajouter un attribut temporaire de type ValueIndex, necessaire pour produire
	// les rapports de statistiques detaillées.

	int current = 0;

	KWClass* predictorClass = trainedPredictor->GetPredictorClass();

	KWAttribute* attribute = predictorClass->GetHeadAttribute();

	while (attribute != NULL) {

		if (current % 100 == 0)
			TaskProgression::DisplayProgression(current * 100 / predictorClass->GetAttributeNumber());

		if (TaskProgression::IsInterruptionRequested())
			break;

		if (attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::PREPARED_ATTRIBUTE_METADATA)) {

			if (attribute->GetStructureName() != "DataGrid") {
				// dico trafiqué manuellement avant une évaluation ?
				cout << "Invalid metadata " << KMPredictor::PREPARED_ATTRIBUTE_METADATA << ", for attribute " << attribute->GetName()
					<< " : this prepared attribute should be a DataGrid" << endl;
				predictorClass->GetNextAttribute(attribute);
				continue;
			}

			current++;
			TaskProgression::DisplayLabel("Modeling dictionary generation : adding cell index attribute " + ALString(IntToString(current)));

			// Ajout de l'attribut natif comme operande
			KWAttribute* nativeAttribute = predictorClass->LookupAttribute(attribute->GetConstMetaData()->GetStringValueAt(KMPredictor::PREPARED_ATTRIBUTE_METADATA));

			if (nativeAttribute == NULL) {
				// dico trafiqué manuellement avant une évaluation ?
				cout << "Invalid metadata " << KMPredictor::PREPARED_ATTRIBUTE_METADATA << ", for attribute " << attribute->GetName()
					<< " : unknown native attribute name '" << attribute->GetConstMetaData()->GetStringValueAt(KMPredictor::PREPARED_ATTRIBUTE_METADATA) << "'" << endl;
				predictorClass->GetNextAttribute(attribute);
				continue;
			}

			// Creation d'une regle pour indexer les cellules
			KWDRCellIndex* valueIndexRule = new KWDRCellIndex;
			valueIndexRule->GetFirstOperand()->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
			valueIndexRule->GetFirstOperand()->SetAttributeName(attribute->GetName());

			valueIndexRule->DeleteAllVariableOperands();
			KWDerivationRuleOperand* operand = new KWDerivationRuleOperand;
			valueIndexRule->AddOperand(operand);
			operand->SetOrigin(KWDerivationRuleOperand::OriginAttribute);
			operand->SetType(nativeAttribute->GetType());
			operand->SetAttributeName(nativeAttribute->GetName());
			valueIndexRule->CompleteTypeInfo(predictorClass);

			const double level = attribute->GetConstMetaData()->GetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey());

			// Ajout de l'attribut de calcul des index de valeurs cibles
			KWAttribute* valueIndexAttribute = new KWAttribute;
			valueIndexAttribute->SetName(predictorClass->BuildAttributeName("CellIndex" + attribute->GetName()));
			valueIndexAttribute->SetDerivationRule(valueIndexRule);
			valueIndexAttribute->SetType(valueIndexAttribute->GetDerivationRule()->GetType());
			valueIndexAttribute->GetMetaData()->SetNoValueAt(KMPredictor::CELL_INDEX_METADATA);
			valueIndexAttribute->GetMetaData()->SetDoubleValueAt(KWDataPreparationAttribute::GetLevelMetaDataKey(), level);
			valueIndexAttribute->CompleteTypeInfo(predictorClass);
			valueIndexAttribute->SetName(predictorClass->BuildAttributeName(valueIndexAttribute->GetName()));
			predictorClass->InsertAttribute(valueIndexAttribute);
		}

		predictorClass->GetNextAttribute(attribute);
	}

	predictorClass->Compile();

	TaskProgression::DisplayLabel("");
}
