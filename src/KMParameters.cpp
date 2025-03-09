// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMParameters.h"
#include "KMParametersView.h"


KMParameters::KMParameters()
{
	// valeurs par defaut
	nMaxIterations = 0;
	nBisectingMaxIterations = 0;
	bSupervisedMode = false;
	iKValue = K_DEFAULT_VALUE;
	iMinKValuePostOptimization = 1;
	distanceType = L2Norm;
	clusteringType = KMeans;
	centroidType = CentroidVirtual;
	clustersCentersInitMethod = ClustersCentersInitMethod::ClustersCentersInitMethodAutomaticallyComputed;
	categoricalPreprocessingType = PreprocessingType::AutomaticallyComputed;
	continuousPreprocessingType = PreprocessingType::AutomaticallyComputed;
	nEpsilonMaxIterations = EPSILON_MAX_ITERATIONS_DEFAULT_VALUE;
	dEpsilonValue = EPSILON_DEFAULT_VALUE;
	iPreprocessingMaxIntervalNumber = PREPROCESSING_MAX_INTERVAL_DEFAULT_VALUE;
	iPreprocessingMaxGroupNumber = PREPROCESSING_MAX_GROUP_DEFAULT_VALUE;
	iPreprocessingSupervisedMaxIntervalNumber = 0;
	iPreprocessingSupervisedMaxGroupNumber = 0;
	iLearningNumberOfReplicates = REPLICATE_NUMBER_DEFAULT_VALUE;
	iMiniBatchSize = MINI_BATCH_SIZE_DEFAULT_VALUE;
	iPostOptimizationVnsLevel = 0;
	iBisectingNumberOfReplicates = REPLICATE_NUMBER_DEFAULT_VALUE;
	replicateChoice = ReplicateChoice::ReplicateChoiceAutomaticallyComputed;
	localModelType = LocalModelType::None;
	bVerboseMode = false;
	bParallelMode = false;
	bMiniBatchMode = false;
	replicatePostOptimization = NoOptimization;
	bBisectingVerboseMode = false;
	bWriteDetailedStatistics = true;
	bLocalModelUseMODL = true;
	iMaxEvaluatedAttributesNumber = 0;
	idClusterAttribute = NULL;
	bKeepNulLevelVariables = false;
}


KMParameters::~KMParameters()
{
	odKMeanAttributesNames.DeleteAll();
	odLoadedAttributesNames.DeleteAll();
	odRecodedAttributesNames.DeleteAll();
}

void KMParameters::AddAttributes(const KWClass* kwc) {

	// memoriser les attributs chargés, à partir d'un dico

	require(kwc != NULL);
	require(kwc->Check());

	KWLoadIndex invalidLoadIndex;

	odKMeanAttributesNames.DeleteAll();
	odLoadedAttributesNames.DeleteAll();

	livKMeanAttributesLoadIndexes.SetSize(0);
	livNativeAttributesLoadIndexes.SetSize(0);
	livLoadedAttributesLoadIndexes.SetSize(0);

	for (int attr = 0; attr < kwc->GetLoadedAttributeNumber(); attr++) {

		livLoadedAttributesLoadIndexes.Add(kwc->GetLoadedAttributeAt(attr)->GetLoadIndex());

		const KWAttribute* attribute = kwc->GetLoadedAttributeAt(attr);

		IntObject* io = new IntObject;
		io->SetInt(livLoadedAttributesLoadIndexes.GetSize() - 1); // pouvoir retrouver indirectement l'index de chargement de l'attribut (car helas, KWLoadIndex n'herite pas de Object, donc pas possible de mettre un KWLoadIndex * comme valeur)
		odLoadedAttributesNames.SetAt(attribute->GetName(), io);// liste de tous les attributs chargés

		if (attribute->GetConstMetaData()->IsKeyPresent(KM_ATTRIBUTE_LABEL)) {
			odKMeanAttributesNames.SetAt(attribute->GetName(), io->Clone());
			livKMeanAttributesLoadIndexes.Add(attribute->GetLoadIndex());
		}
		else
			livKMeanAttributesLoadIndexes.Add(invalidLoadIndex);

		if (attribute->GetConstMetaData()->IsKeyPresent(SELECTED_NATIVE_ATTRIBUTE_LABEL))
			livNativeAttributesLoadIndexes.Add(attribute->GetLoadIndex());
		else
			livNativeAttributesLoadIndexes.Add(invalidLoadIndex);

	}
}

void KMParameters::AddRecodedAttribute(const KWAttribute* nativeAttribute, const KWAttribute* recodedAttribute) {

	// garder la trace de la correspondance entre attribut natif et attribut recodé, aux fins d'affichage dans les rapports

	require(nativeAttribute != NULL);
	require(recodedAttribute != NULL);

	Object* o = odRecodedAttributesNames.Lookup(recodedAttribute->GetName());

	if (o == NULL) {
		// nouvelle entree
		StringObject* so = new StringObject;
		so->SetString(nativeAttribute->GetName());
		odRecodedAttributesNames.SetAt(recodedAttribute->GetName(), so);
	}
	else {
		// mise a jour entree existante
		StringObject* so = cast(StringObject*, o);
		so->SetString(nativeAttribute->GetName());
	}
}

ALString KMParameters::GetLoadedAttributeNameByRank(const int idx) const {

	POSITION position = odLoadedAttributesNames.GetStartPosition();
	ALString key;
	Object* oIdx;
	IntObject* i;

	while (position != NULL)
	{
		odLoadedAttributesNames.GetNextAssoc(position, key, oIdx);
		i = cast(IntObject*, oIdx);
		if (i->GetInt() == idx)
			return key;
	}
	return "no attribute loaded at " + ALString(IntToString(idx));

}
ALString KMParameters::GetNativeAttributeName(const ALString recodedAttributeName) const {

	Object* nativeName = odRecodedAttributesNames.Lookup(recodedAttributeName);

	if (nativeName == NULL)
		return "";

	StringObject* soNativeName = cast(StringObject*, nativeName);

	return soNativeName->GetString();

}

void KMParameters::SetIdClusterAttributeFromClass(const KWClass* kwc) {

	assert(kwc != NULL);
	idClusterAttribute = NULL;

	KWAttribute* attribute = kwc->GetHeadAttribute();

	while (attribute != NULL)
	{
		if (attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::ID_CLUSTER_METADATA)) {
			idClusterAttribute = attribute;
			break;
		}

		kwc->GetNextAttribute(attribute);
	}
}


const ObjectDictionary& KMParameters::GetKMAttributeNames() const {
	return odKMeanAttributesNames;
}

const ObjectDictionary& KMParameters::GetLoadedAttributesNames() const {
	return odLoadedAttributesNames;
}

KMParameters* KMParameters::Clone() const
{
	KMParameters* aClone;

	aClone = new KMParameters;
	aClone->CopyFrom(this);
	return aClone;
}

void KMParameters::CopyFrom(const KMParameters* aSource)
{
	require(aSource != NULL);

	nMaxIterations = aSource->nMaxIterations;
	nBisectingMaxIterations = aSource->nBisectingMaxIterations;
	bSupervisedMode = aSource->bSupervisedMode;
	bVerboseMode = aSource->bVerboseMode;
	bParallelMode = aSource->bParallelMode;
	bMiniBatchMode = aSource->bMiniBatchMode;
	replicatePostOptimization = aSource->replicatePostOptimization;
	bBisectingVerboseMode = aSource->bBisectingVerboseMode;
	bWriteDetailedStatistics = aSource->bWriteDetailedStatistics;
	bLocalModelUseMODL = aSource->bLocalModelUseMODL;
	iMaxEvaluatedAttributesNumber = aSource->iMaxEvaluatedAttributesNumber;
	iKValue = aSource->iKValue;
	iMinKValuePostOptimization = aSource->iMinKValuePostOptimization;
	asMainTargetModality = aSource->asMainTargetModality;
	distanceType = aSource->distanceType;
	clusteringType = aSource->clusteringType;
	centroidType = aSource->centroidType;
	clustersCentersInitMethod = aSource->clustersCentersInitMethod;
	categoricalPreprocessingType = aSource->categoricalPreprocessingType;
	continuousPreprocessingType = aSource->continuousPreprocessingType;
	nEpsilonMaxIterations = aSource->nEpsilonMaxIterations;
	dEpsilonValue = aSource->dEpsilonValue;
	iPreprocessingMaxIntervalNumber = aSource->iPreprocessingMaxIntervalNumber;
	iPreprocessingMaxGroupNumber = aSource->iPreprocessingMaxGroupNumber;
	iPreprocessingSupervisedMaxIntervalNumber = aSource->iPreprocessingSupervisedMaxIntervalNumber;
	iPreprocessingSupervisedMaxGroupNumber = aSource->iPreprocessingSupervisedMaxGroupNumber;
	iLearningNumberOfReplicates = aSource->iLearningNumberOfReplicates;
	iMiniBatchSize = aSource->iMiniBatchSize;
	iPostOptimizationVnsLevel = aSource->iPostOptimizationVnsLevel;
	iBisectingNumberOfReplicates = aSource->iBisectingNumberOfReplicates;
	replicateChoice = aSource->replicateChoice;
	localModelType = aSource->localModelType;
	livKMeanAttributesLoadIndexes.CopyFrom(&aSource->livKMeanAttributesLoadIndexes);
	livNativeAttributesLoadIndexes.CopyFrom(&aSource->livNativeAttributesLoadIndexes);
	bKeepNulLevelVariables = aSource->bKeepNulLevelVariables;

	// copier les StringObject de odRecodedAttributesNames
	odRecodedAttributesNames.DeleteAll();
	POSITION position = aSource->odRecodedAttributesNames.GetStartPosition();
	ALString key;
	Object* oIdx;
	StringObject* so;
	while (position != NULL)
	{
		aSource->odRecodedAttributesNames.GetNextAssoc(position, key, oIdx);
		so = cast(StringObject*, oIdx);
		StringObject* newString = new StringObject;
		newString->SetString(so->GetString());
		odRecodedAttributesNames.SetAt(key, newString);
	}

	// copier les IntObject de odKMeanAttributesNames
	odKMeanAttributesNames.DeleteAll();
	position = aSource->odKMeanAttributesNames.GetStartPosition();
	IntObject* io;
	while (position != NULL)
	{
		aSource->odKMeanAttributesNames.GetNextAssoc(position, key, oIdx);
		io = cast(IntObject*, oIdx);
		IntObject* newInteger = new IntObject;
		newInteger->SetInt(io->GetInt());
		odKMeanAttributesNames.SetAt(key, newInteger);
	}

	// copier les IntObject de odLoadedAttributesNames
	odLoadedAttributesNames.DeleteAll();
	position = aSource->odLoadedAttributesNames.GetStartPosition();
	while (position != NULL)
	{
		aSource->odLoadedAttributesNames.GetNextAssoc(position, key, oIdx);
		io = cast(IntObject*, oIdx);
		IntObject* newInteger = new IntObject;
		newInteger->SetInt(io->GetInt());
		odLoadedAttributesNames.SetAt(key, newInteger);
	}
}

const ALString KMParameters::GetDistanceTypeLabel() const
{
	switch (distanceType) {

	case L1Norm:
		return KMParametersView::L1_NORM_LABEL;
		break;

	case L2Norm:
		return KMParametersView::L2_NORM_LABEL;
		break;

	case CosineNorm:
		return KMParametersView::COSINUS_NORM_LABEL;
		break;

	default:
		return "undefined";

	}
}

const ALString KMParameters::GetCentroidTypeLabel() const
{
	switch (centroidType) {

	case CentroidRealInstance:
		return KMParameters::CENTROID_REAL_INSTANCE_LABEL;
		break;

	case CentroidVirtual:
		return KMParameters::CENTROID_VIRTUAL_LABEL;
		break;

	default:
		return "undefined";

	}
}
const ALString KMParameters::GetReplicateChoiceLabel() const
{
	switch (replicateChoice) {

	case Distance:
		return REPLICATE_DISTANCE_LABEL;
		break;

	case EVA:
		return REPLICATE_EVA_LABEL;
		break;

	case ARIByClusters:
		return REPLICATE_ARI_BY_CLUSTERS_LABEL;
		break;

	case ARIByClasses:
		return REPLICATE_ARI_BY_CLASSES_LABEL;
		break;

	case NormalizedMutualInformationByClusters:
		return REPLICATE_NORMALIZED_MUTUAL_INFORMATION_BY_CLUSTERS_LABEL;
		break;

	case NormalizedMutualInformationByClasses:
		return REPLICATE_NORMALIZED_MUTUAL_INFORMATION_BY_CLASSES_LABEL;
		break;

	case LEVA:
		return REPLICATE_LEVA_LABEL;
		break;

	case DaviesBouldin:
		return REPLICATE_DAVIES_BOULDIN_LABEL;
		break;

	case VariationOfInformation:
		return REPLICATE_VARIATION_OF_INFORMATION_LABEL;
		break;

	case PredictiveClustering:
		return REPLICATE_PREDICTIVE_CLUSTERING_LABEL;
		break;

	case ReplicateChoiceAutomaticallyComputed:
		return AUTO_COMPUTED_LABEL;
		break;

	default:
		return "undefined";

	}
}


const ALString KMParameters::GetReplicatePostOptimizationLabel() const
{
	switch (replicatePostOptimization) {

	case NoOptimization:
		return KMParametersView::NONE_LABEL;
		break;

	case FastOptimization:
		return KMParametersView::REPLICATE_POST_OPTIMIZATION_FAST_LABEL;
		break;

	default:
		return "undefined";

	}
}

const ALString KMParameters::GetLocalModelTypeLabel() const
{
	switch (localModelType) {

	case None:
		return KMParametersView::NONE_LABEL;
		break;

	case SNB:
		return KMParametersView::LOCAL_MODEL_SNB_LABEL;
		break;

	case NB:
		return KMParametersView::LOCAL_MODEL_NB_LABEL;
		break;

	default:
		return "undefined";

	}
}


const ALString KMParameters::GetClustersCentersInitializationMethodLabel() const
{
	switch (clustersCentersInitMethod) {

	case ClustersCentersInitMethod::ClustersCentersInitMethodAutomaticallyComputed:
		return AUTO_COMPUTED_LABEL;

	case ClustersCentersInitMethod::Random:
		return KMParametersView::RANDOM_LABEL;

	case ClustersCentersInitMethod::Sample:
		return KMParametersView::SAMPLE_LABEL;

	case ClustersCentersInitMethod::KMeanPlusPlus:
		return KMParametersView::KMEAN_PLUS_PLUS_LABEL;

	case ClustersCentersInitMethod::KMeanPlusPlusR:
		return KMParametersView::KMEAN_PLUS_PLUS_R_LABEL;

	case ClustersCentersInitMethod::RocchioThenSplit:
		return KMParametersView::ROCCHIO_SPLIT_LABEL;

	case ClustersCentersInitMethod::Bisecting:
		return KMParametersView::BISECTING_LABEL;

	case ClustersCentersInitMethod::MinMaxRandom:
		return KMParametersView::MIN_MAX_RANDOM_LABEL;

	case ClustersCentersInitMethod::MinMaxDeterministic:
		return KMParametersView::MIN_MAX_DETERMINISTIC_LABEL;

	case ClustersCentersInitMethod::VariancePartitioning:
		return KMParametersView::PCA_PART_LABEL;

	case ClustersCentersInitMethod::ClassDecomposition:
		return KMParametersView::CLASS_DECOMPOSITION_LABEL;

	default:
		return "undefined";

	}
}
const ALString KMParameters::GetCategoricalPreprocessingTypeLabel(const bool bTranslateAutomaticallyComputed) const
{
	switch (categoricalPreprocessingType) {

	case PreprocessingType::UnusedVariable:
		return UNUSED_VARIABLE_LABEL;

	case PreprocessingType::AutomaticallyComputed:

		if (not bTranslateAutomaticallyComputed)
			return AUTO_COMPUTED_LABEL;
		else {
			if (not bSupervisedMode)
				return BASIC_GROUPING_LABEL;
			else
				return SOURCE_CONDITIONAL_INFO_LABEL;
		}

	case PreprocessingType::BasicGrouping:
		return BASIC_GROUPING_LABEL;

	case PreprocessingType::Binarization:
		return BINARIZATION_LABEL;

	case PreprocessingType::HammingConditionalInfo:
		return HAMMING_CONDITIONAL_INFO_CATEGORICAL_LABEL;

	case PreprocessingType::ConditionaInfoWithPriors:
		return CONDITIONAL_INFO_WITH_PRIORS_CATEGORICAL_LABEL;

	case PreprocessingType::Entropy:
		return ENTROPY_CATEGORICAL_LABEL;

	case PreprocessingType::EntropyWithPriors:
		return ENTROPY_WITH_PRIORS_CATEGORICAL_LABEL;

	default:
		return "undefined";
	}
}

void KMParameters::SetCategoricalPreprocessingType(const ALString preprocessingTypeLabel)
{
	if (preprocessingTypeLabel == UNUSED_VARIABLE_LABEL)
		categoricalPreprocessingType = PreprocessingType::UnusedVariable;
	else
		if (preprocessingTypeLabel == AUTO_COMPUTED_LABEL)
			categoricalPreprocessingType = PreprocessingType::AutomaticallyComputed;
		else
			if (preprocessingTypeLabel == BINARIZATION_LABEL)
				categoricalPreprocessingType = PreprocessingType::Binarization;
			else
				if (preprocessingTypeLabel == HAMMING_CONDITIONAL_INFO_CATEGORICAL_LABEL)
					categoricalPreprocessingType = PreprocessingType::HammingConditionalInfo;
				else
					if (preprocessingTypeLabel == BASIC_GROUPING_LABEL)
						categoricalPreprocessingType = PreprocessingType::BasicGrouping;
					else
						if (preprocessingTypeLabel == CONDITIONAL_INFO_WITH_PRIORS_CATEGORICAL_LABEL)
							categoricalPreprocessingType = PreprocessingType::ConditionaInfoWithPriors;
						else
							if (preprocessingTypeLabel == ENTROPY_CATEGORICAL_LABEL)
								categoricalPreprocessingType = PreprocessingType::Entropy;
							else
								if (preprocessingTypeLabel == ENTROPY_WITH_PRIORS_CATEGORICAL_LABEL)
									categoricalPreprocessingType = PreprocessingType::EntropyWithPriors;
								else
									if (preprocessingTypeLabel == SOURCE_CONDITIONAL_INFO_LABEL)
										categoricalPreprocessingType = PreprocessingType::SourceConditionalInfo;
									else
										if (preprocessingTypeLabel == AUTO_COMPUTED_LABEL)
											categoricalPreprocessingType = PreprocessingType::AutomaticallyComputed;
										else
											// si valeur invalide transmise via un script batch
											AddError("Invalid value for CategoricalPreprocessingType : '" + preprocessingTypeLabel + "'. Beware that labels are case-sensitive.");
}

const ALString KMParameters::GetContinuousPreprocessingTypeLabel(const bool bTranslateAutomaticallyComputed) const
{
	switch (continuousPreprocessingType) {

	case PreprocessingType::UnusedVariable:
		return UNUSED_VARIABLE_LABEL;

	case PreprocessingType::NoPreprocessing:
		return NO_PREPROCESSING_LABEL;

	case PreprocessingType::AutomaticallyComputed:

		if (not bTranslateAutomaticallyComputed)
			return AUTO_COMPUTED_LABEL;
		else {
			if (not bSupervisedMode)
				return RANK_NORMALIZATION_LABEL;
			else
				return SOURCE_CONDITIONAL_INFO_LABEL;
		}

	case PreprocessingType::CenterReduction:
		return CENTER_REDUCTION_LABEL;

	case PreprocessingType::Binarization:
		return BINARIZATION_LABEL;

	case PreprocessingType::RankNormalization:
		return RANK_NORMALIZATION_LABEL;

	case PreprocessingType::Normalization:
		return NORMALIZATION_LABEL;

	case PreprocessingType::ConditionaInfoWithPriors:
		return CONDITIONAL_INFO_WITH_PRIORS_CONTINUOUS_LABEL;

	case PreprocessingType::Entropy:
		return ENTROPY_CONTINUOUS_LABEL;

	case PreprocessingType::EntropyWithPriors:
		return ENTROPY_WITH_PRIORS_CONTINUOUS_LABEL;

	case PreprocessingType::HammingConditionalInfo:
		return HAMMING_CONDITIONAL_INFO_CONTINUOUS_LABEL;

	default:
		return "undefined";
	}
}

void KMParameters::SetContinuousPreprocessingType(const ALString preprocessingTypeLabel)
{

	if (preprocessingTypeLabel == UNUSED_VARIABLE_LABEL)
		continuousPreprocessingType = PreprocessingType::UnusedVariable;
	else
		if (preprocessingTypeLabel == NO_PREPROCESSING_LABEL)
			continuousPreprocessingType = PreprocessingType::NoPreprocessing;
		else
			if (preprocessingTypeLabel == AUTO_COMPUTED_LABEL)
				continuousPreprocessingType = PreprocessingType::AutomaticallyComputed;
			else
				if (preprocessingTypeLabel == CENTER_REDUCTION_LABEL)
					continuousPreprocessingType = PreprocessingType::CenterReduction;
				else
					if (preprocessingTypeLabel == RANK_NORMALIZATION_LABEL)
						continuousPreprocessingType = PreprocessingType::RankNormalization;
					else
						if (preprocessingTypeLabel == NORMALIZATION_LABEL)
							continuousPreprocessingType = PreprocessingType::Normalization;
						else
							if (preprocessingTypeLabel == BINARIZATION_LABEL)
								continuousPreprocessingType = PreprocessingType::Binarization;
							else
								if (preprocessingTypeLabel == HAMMING_CONDITIONAL_INFO_CONTINUOUS_LABEL)
									continuousPreprocessingType = PreprocessingType::HammingConditionalInfo;
								else
									if (preprocessingTypeLabel == CONDITIONAL_INFO_WITH_PRIORS_CONTINUOUS_LABEL)
										continuousPreprocessingType = PreprocessingType::ConditionaInfoWithPriors;
									else
										if (preprocessingTypeLabel == ENTROPY_CONTINUOUS_LABEL)
											continuousPreprocessingType = PreprocessingType::Entropy;
										else
											if (preprocessingTypeLabel == ENTROPY_WITH_PRIORS_CONTINUOUS_LABEL)
												continuousPreprocessingType = PreprocessingType::EntropyWithPriors;
											else
												if (preprocessingTypeLabel == SOURCE_CONDITIONAL_INFO_LABEL)
													continuousPreprocessingType = PreprocessingType::SourceConditionalInfo;
												else
													if (preprocessingTypeLabel == AUTO_COMPUTED_LABEL)
														continuousPreprocessingType = PreprocessingType::AutomaticallyComputed;
													else
														// si valeur invalide transmise via un script batch
														AddError("Invalid value for ContinuousPreprocessingType : '" + preprocessingTypeLabel + "'. Beware that labels are case-sensitive.");
}

void KMParameters::SetClustersCentersInitializationMethod(const ALString label)
{
	if (label == KMParameters::AUTO_COMPUTED_LABEL)
		clustersCentersInitMethod = KMParameters::ClustersCentersInitMethod::ClustersCentersInitMethodAutomaticallyComputed;
	else
		if (label == KMParametersView::RANDOM_LABEL)
			clustersCentersInitMethod = KMParameters::Random;
		else
			if (label == KMParametersView::SAMPLE_LABEL)
				clustersCentersInitMethod = KMParameters::Sample;
			else
				if (label == KMParametersView::ROCCHIO_SPLIT_LABEL)
					clustersCentersInitMethod = KMParameters::RocchioThenSplit;
				else
					if (label == KMParametersView::KMEAN_PLUS_PLUS_LABEL)
						clustersCentersInitMethod = KMParameters::KMeanPlusPlus;
					else
						if (label == KMParametersView::KMEAN_PLUS_PLUS_R_LABEL)
							clustersCentersInitMethod = KMParameters::KMeanPlusPlusR;
						else
							if (label == KMParametersView::BISECTING_LABEL)
								clustersCentersInitMethod = KMParameters::Bisecting;
							else
								if (label == KMParametersView::MIN_MAX_RANDOM_LABEL)
									clustersCentersInitMethod = KMParameters::MinMaxRandom;
								else
									if (label == KMParametersView::MIN_MAX_DETERMINISTIC_LABEL)
										clustersCentersInitMethod = KMParameters::MinMaxDeterministic;
									else
										if (label == KMParametersView::PCA_PART_LABEL)
											clustersCentersInitMethod = KMParameters::VariancePartitioning;
										else
											if (label == KMParametersView::CLASS_DECOMPOSITION_LABEL)
												clustersCentersInitMethod = KMParameters::ClassDecomposition;
											else
												// si valeur invalide transmise via un script batch
												AddError("Invalid value for CentersInitializationMethod : '" + label + "'. Beware that labels are case-sensitive.");


}
void KMParameters::SetCentroidType(const ALString label) {

	if (label == KMParameters::CENTROID_VIRTUAL_LABEL)
		centroidType = KMParameters::CentroidVirtual;
	else
		if (label == KMParameters::CENTROID_REAL_INSTANCE_LABEL)
			centroidType = KMParameters::CentroidRealInstance;
		else
			// si valeur invalide transmise via un script batch
			AddError("Invalid value for CentroidType : '" + label + "'. Beware that labels are case-sensitive.");

}

void KMParameters::SetDistanceType(const ALString label) {

	if (label == KMParametersView::L2_NORM_LABEL)
		distanceType = KMParameters::L2Norm;
	else
		if (label == KMParametersView::L1_NORM_LABEL)
			distanceType = KMParameters::L1Norm;
		else
			if (label == KMParametersView::COSINUS_NORM_LABEL)
				distanceType = KMParameters::CosineNorm;
			else
				// si valeur invalide transmise via un script batch
				AddError("Invalid value for DistanceType : '" + label + "'. Beware that labels are case-sensitive.");

}

void KMParameters::SetReplicateChoice(const ALString label) {

	if (label == KMParameters::AUTO_COMPUTED_LABEL)
		replicateChoice = ReplicateChoice::ReplicateChoiceAutomaticallyComputed;
	else
		if (label == KMParameters::REPLICATE_DISTANCE_LABEL)
			replicateChoice = ReplicateChoice::Distance;
		else
			if (label == KMParameters::REPLICATE_ARI_BY_CLUSTERS_LABEL)
				replicateChoice = ReplicateChoice::ARIByClusters;
			else
				if (label == KMParameters::REPLICATE_NORMALIZED_MUTUAL_INFORMATION_BY_CLUSTERS_LABEL)
					replicateChoice = ReplicateChoice::NormalizedMutualInformationByClusters;
				else
					if (label == KMParameters::REPLICATE_NORMALIZED_MUTUAL_INFORMATION_BY_CLASSES_LABEL)
						replicateChoice = ReplicateChoice::NormalizedMutualInformationByClasses;
					else
						if (label == KMParameters::REPLICATE_ARI_BY_CLASSES_LABEL)
							replicateChoice = ReplicateChoice::ARIByClasses;
						else
							if (label == KMParameters::REPLICATE_LEVA_LABEL)
								replicateChoice = ReplicateChoice::LEVA;
							else
								if (label == KMParameters::REPLICATE_DAVIES_BOULDIN_LABEL)
									replicateChoice = ReplicateChoice::DaviesBouldin;
								else
									if (label == KMParameters::REPLICATE_VARIATION_OF_INFORMATION_LABEL)
										replicateChoice = ReplicateChoice::VariationOfInformation;
									else
										if (label == KMParameters::REPLICATE_PREDICTIVE_CLUSTERING_LABEL)
											replicateChoice = ReplicateChoice::PredictiveClustering;
										else
											if (label == KMParameters::REPLICATE_EVA_LABEL)
												replicateChoice = ReplicateChoice::EVA;
											else
												// si valeur invalide transmise via un script batch
												AddError("Invalid value for ReplicateChoice : '" + label + "'. Beware that labels are case-sensitive.");
}


void KMParameters::SetLocalModelType(const ALString label) {

	if (label == KMParametersView::NONE_LABEL)
		localModelType = LocalModelType::None;
	else
		if (label == KMParametersView::LOCAL_MODEL_NB_LABEL)
			localModelType = LocalModelType::NB;
		else
			if (label == KMParametersView::LOCAL_MODEL_SNB_LABEL)
				localModelType = LocalModelType::SNB;
			else
				// si valeur invalide transmise via un script batch
				AddError("Invalid value for LocalModelType : '" + label + "'. Beware that labels are case-sensitive.");

}


void KMParameters::SetReplicatePostOptimization(const ALString label) {

	if (label == KMParametersView::NONE_LABEL)
		replicatePostOptimization = NoOptimization;
	else
		if (label == KMParametersView::REPLICATE_POST_OPTIMIZATION_FAST_LABEL)
			replicatePostOptimization = FastOptimization;
		else
			// si valeur invalide transmise via un script batch
			AddError("Invalid value for ReplicatePostOptimization : '" + label + "'. Beware that labels are case-sensitive.");
}

void KMParameters::Write(ostream& ost) const
{
	ost << endl << "K input value: " + ALString(IntToString(GetKValue()));
	ost << endl << "Min K value for post-optimisation training: " + ALString(IntToString(GetMinKValuePostOptimization()));
	ost << endl << "Local models = " + GetLocalModelTypeLabel();
	ost << endl << "Always use MODL for preprocessing in local models : " + ALString((bLocalModelUseMODL ? "yes" : "no"));

	if (bSupervisedMode)
		ost << endl << "Max number of used variables: " << GetMaxEvaluatedAttributesNumber();

	ost << endl << "Number of replicates: " + ALString(IntToString(GetLearningNumberOfReplicates()));
	ost << endl << "Best replicate selection: " + ALString(GetReplicateChoiceLabel());
	ost << endl << "Best replicate post-optimization: " + ALString(GetReplicatePostOptimizationLabel());
	ost << endl << "VNS optimization level: " + ALString(IntToString(GetPostOptimizationVnsLevel()));
	ost << endl << "Continuous preprocessing: " + ALString(GetContinuousPreprocessingTypeLabel(true));
	ost << endl << "Categorical preprocessing: " + ALString(GetCategoricalPreprocessingTypeLabel(true));
	ost << endl << "Clusters initialization: " + ALString(GetClustersCentersInitializationMethodLabel());

	if (GetLearningExpertMode()) {

		ost << endl << "Mini-batches mode: " + ALString((bMiniBatchMode ? "yes" : "no"));
		if (bMiniBatchMode)
			ost << endl << "Number of instances in each mini-batch: " + ALString(IntToString(GetMiniBatchSize()));
		ost << endl << "Max iterations number: " + ALString(IntToString(GetMaxIterations()));

		if (bSupervisedMode) {
			ost << endl << "Pre-processing max intervals : " << GetPreprocessingSupervisedMaxIntervalNumber();
			ost << endl << "Pre-processing max groups: " << GetPreprocessingSupervisedMaxGroupNumber();
		}
		else
		{
			ost << endl << "Pre-processing max intervals (rank normalization) : " << GetPreprocessingMaxIntervalNumber();
			ost << endl << "Pre-processing max groups (basic grouping): " << GetPreprocessingMaxGroupNumber();
		}
		ost << endl << "Epsilon value: " + KMGetDisplayString(GetEpsilonValue());
		ost << endl << "Max epsilon iterations number: " + ALString(IntToString(GetEpsilonMaxIterations()));
		ost << endl << "Centroid type: " + ALString(GetCentroidTypeLabel());
		ost << endl << "Bisecting/class decomposition number of replicates: " + ALString(IntToString(GetBisectingNumberOfReplicates()));
		ost << endl << "Bisecting/class decomposition iterations max number: " + ALString(IntToString(GetBisectingMaxIterations()));
	}

	ost << endl << "Distance norm: " + ALString(GetDistanceTypeLabel());

	if (bSupervisedMode) {
		ost << endl << "Keep null level variables in case of unsupervised preprocessing: " + ALString((bKeepNulLevelVariables ? "yes" : "no"));
	}
	ost << endl;
}

void KMParameters::WriteJSON(JSONFile* fJSON) const
{
	// on ecrit uniquement le mode non expert
	fJSON->BeginKeyObject("parameters");
	fJSON->WriteKeyInt("kInputValue", GetKValue());
	fJSON->WriteKeyInt("minKpostOptimization", GetMinKValuePostOptimization());
	fJSON->WriteKeyString("clustersInitialization", ALString(GetClustersCentersInitializationMethodLabel()));
	fJSON->WriteKeyString("continuousPreprocessing", ALString(GetContinuousPreprocessingTypeLabel(true)));
	fJSON->WriteKeyString("categoricalPreprocessing", ALString(GetCategoricalPreprocessingTypeLabel(true)));
	fJSON->WriteKeyString("replicateSelection", ALString(GetReplicateChoiceLabel()));
	fJSON->WriteKeyString("replicatePostOptimization", ALString(GetReplicatePostOptimizationLabel()));
	fJSON->WriteKeyInt("numberOfReplicates", GetLearningNumberOfReplicates());
	fJSON->WriteKeyString("localModelType", ALString(GetLocalModelTypeLabel()));
	fJSON->WriteKeyInt("vnsOptimizationLevel", GetPostOptimizationVnsLevel());
	if (bSupervisedMode) {
		fJSON->WriteKeyInt("maxNumberOfUsedVariables", GetMaxEvaluatedAttributesNumber());
		fJSON->WriteKeyString("keepNullLevelVariablesUnsupervisedPreprocessing", ALString((bKeepNulLevelVariables ? "yes" : "no")));
	}
	fJSON->WriteKeyString("distanceType", ALString(GetDistanceTypeLabel()));

	fJSON->EndObject();
}


const ALString KMParameters::GetClassLabel() const
{
	return "Selection parameters";
}


void KMParameters::PrepareDeploymentClass(KWClass* modelingClass) {

	KWAttribute* attribute = modelingClass->GetHeadAttribute();

	while (attribute != NULL) {

		// charger les attributs supplementaires indispensables a l'evaluation
		if (attribute->GetConstMetaData()->IsKeyPresent(KM_ATTRIBUTE_LABEL) or
			attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::ID_CLUSTER_METADATA) or
			attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::DISTANCE_CLUSTER_LABEL)) {

			attribute->SetUsed(true);
			attribute->SetLoaded(true);
		}
		else {

			if (bWriteDetailedStatistics) { //charger les attributs necessaires a la production des stats detaillees (frequences de modalites par cluster)

				if (attribute->GetConstMetaData()->IsKeyPresent(SELECTED_NATIVE_ATTRIBUTE_LABEL) or
					attribute->GetConstMetaData()->IsKeyPresent(KMPredictor::CELL_INDEX_METADATA)) {

					attribute->SetUsed(true);
					attribute->SetLoaded(true);
				}
			}
		}

		modelingClass->GetNextAttribute(attribute);
	}

	modelingClass->Compile();

	// renseigner les differentes structures permettant de retrouver les informations liées aux attributs
	// ce sont notamment les index de chargements, memorises pour des questions de performance
	AddAttributes(modelingClass);
}

const ALString KMParameters::GetObjectLabel() const
{
	return "KMParameters";
}

const boolean  KMParameters::GetSupervisedMode() const {
	return bSupervisedMode;
}

void  KMParameters::SetSupervisedMode(boolean b) {
	bSupervisedMode = b;
}

const boolean  KMParameters::GetVerboseMode() const {
	return bVerboseMode;
}
const boolean  KMParameters::GetParallelMode() const {
	return bParallelMode;
}
void  KMParameters::SetMiniBatchMode(boolean b) {
	bMiniBatchMode = b;
}
const boolean  KMParameters::GetMiniBatchMode() const {
	return bMiniBatchMode;
}
void  KMParameters::SetVerboseMode(boolean b) {
	bVerboseMode = b;
}
void  KMParameters::SetParallelMode(boolean b) {
	bParallelMode = b;
}
const KMParameters::ReplicatePostOptimization  KMParameters::GetReplicatePostOptimization() const {
	return replicatePostOptimization;
}
void  KMParameters::SetReplicatePostOptimization(KMParameters::ReplicatePostOptimization b) {
	replicatePostOptimization = b;
}

const boolean  KMParameters::GetLocalModelUseMODL() const {
	return bLocalModelUseMODL;
}
void  KMParameters::SetLocalModelUseMODL(boolean b) {
	bLocalModelUseMODL = b;
}

const boolean  KMParameters::GetBisectingVerboseMode() const {
	return bBisectingVerboseMode;
}
void  KMParameters::SetBisectingVerboseMode(boolean b) {
	bBisectingVerboseMode = b;
}
const boolean  KMParameters::GetWriteDetailedStatistics() const {
	return bWriteDetailedStatistics;
}
void  KMParameters::SetWriteDetailedStatistics(boolean b) {
	bWriteDetailedStatistics = b;
}
const int  KMParameters::GetMaxEvaluatedAttributesNumber() const {
	return iMaxEvaluatedAttributesNumber;
}
void  KMParameters::SetMaxEvaluatedAttributesNumber(int i) {
	iMaxEvaluatedAttributesNumber = i;
}
const boolean  KMParameters::GetKeepNulLevelVariables() const {
	return bKeepNulLevelVariables;
}
void  KMParameters::SetKeepNulLevelVariables(boolean b) {
	bKeepNulLevelVariables = b;
}

boolean KMParameters::Check()
{
	boolean bOk = true;

	// si pas de variable cible, passer automatiquement la selection des replicates sur le critère "distance"
	if (replicateChoice == ReplicateChoice::ReplicateChoiceAutomaticallyComputed)
		replicateChoice = (bSupervisedMode ? ARIByClusters : Distance);

	if (clustersCentersInitMethod == ClustersCentersInitMethod::ClustersCentersInitMethodAutomaticallyComputed)
		clustersCentersInitMethod = (bSupervisedMode ? KMeanPlusPlusR : KMeanPlusPlus);

	if (GetMaxIterations() < -1 or GetMaxIterations() > 50) {
		AddError("Max iterations must be lower than 50. Use 0 if no maximum is required.");
		bOk = false;
	}
	if (GetContinuousPreprocessingType() == UnusedVariable and
		GetCategoricalPreprocessingType() == UnusedVariable) {

		AddError("Preprocessing error : continuous and categorical are both unused. Check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and GetReplicateChoice() == EVA) {
		AddError("EVA replicate selection is available only in supervised mode : Check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and
		(GetReplicateChoice() == ARIByClasses or GetReplicateChoice() == ARIByClusters)) {
		AddError("ARI replicate selection is available only in supervised mode : Check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and GetReplicateChoice() == LEVA) {
		AddError("LEVA replicate selection is available only in supervised mode : Check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and GetReplicateChoice() == VariationOfInformation) {
		AddError("Variation of information replicate selection is available only in supervised mode : Check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and GetReplicateChoice() == PredictiveClustering) {
		AddError("Predictive clustering replicate selection is available only in supervised mode : Check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and
		(GetReplicateChoice() == NormalizedMutualInformationByClusters or GetReplicateChoice() == NormalizedMutualInformationByClasses)) {
		AddError("NMI replicate selection is available only in supervised mode : Check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and
		(continuousPreprocessingType == Binarization or categoricalPreprocessingType == Binarization)) {
		AddError("Binarization preprocessing is available only in supervised mode : check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and
		(continuousPreprocessingType == HammingConditionalInfo or categoricalPreprocessingType == HammingConditionalInfo)) {
		AddError("Hamming Conditional Info preprocessing is available only in supervised mode : check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and
		(continuousPreprocessingType == ConditionaInfoWithPriors or categoricalPreprocessingType == ConditionaInfoWithPriors)) {
		AddError("Conditional Info with priors preprocessing is available only in supervised mode : check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and
		(continuousPreprocessingType == Entropy or categoricalPreprocessingType == Entropy)) {
		AddError("Entropy preprocessing is available only in supervised mode : check your parameters.");
		bOk = false;
	}
	if (not bSupervisedMode and
		(continuousPreprocessingType == EntropyWithPriors or categoricalPreprocessingType == EntropyWithPriors)) {
		AddError("Entropy with priors preprocessing is available only in supervised mode : check your parameters.");
		bOk = false;
	}
	if (GetLearningNumberOfReplicates() < 1) {
		AddError("Number of replicates must be > 0.");
		bOk = false;
	}
	if (GetKValue() < 1) {
		AddError("Number of clusters must be > 0.");
		bOk = false;
	}
	if (GetMaxEvaluatedAttributesNumber() < 0) {
		AddError("Max evaluated attributes number must be >= 0.");
		bOk = false;
	}
	if (GetPostOptimizationVnsLevel() < 0) {
		AddError("Post optimization VNS level must be >= 0.");
		bOk = false;
	}

	return bOk;
}

StringObject* KMParameters::GetUniqueLabel(const ObjectArray& existingLabels, const ALString prefix) {

	StringObject* result = new StringObject();
	result->SetString(prefix);

	ALString sSuffix;
	int nIndex = 1;

	while (existingLabels.Lookup(result) != NULL) {

		sSuffix = ALString(" ") + IntToString(nIndex);
		result->SetString(prefix + sSuffix);
		nIndex++;
	}

	return result;
}


int
KMCompareLabels(const void* elem1, const void* elem2)
{
	StringObject* s1 = (StringObject*) * (Object**)elem1;
	StringObject* s2 = (StringObject*) * (Object**)elem2;

	if (s1->GetString() == s2->GetString())
		return 0;
	else
		if (s1->GetString() > s2->GetString())
			return 1;
		else
			return -1;
}

const ALString KMParameters::KM_ATTRIBUTE_LABEL = "KmeansAttribute";
const ALString KMParameters::SELECTED_NATIVE_ATTRIBUTE_LABEL = "SelectedNativeAttribute";
const int KMParameters::K_MAX_VALUE = 50000;
const int KMParameters::REPLICATE_NUMBER_MAX_VALUE = 1000;
const int KMParameters::MINI_BATCH_SIZE_MAX_VALUE = 10000000;
const int KMParameters::K_DEFAULT_VALUE = 1;
const int KMParameters::MAX_ITERATIONS = 1000;
const int KMParameters::EPSILON_MAX_ITERATIONS_DEFAULT_VALUE = 5;
const int KMParameters::EPSILON_MAX_ITERATIONS = 100;
const double KMParameters::EPSILON_DEFAULT_VALUE = 0.000000001;
const int KMParameters::PREPROCESSING_MAX_INTERVAL_DEFAULT_VALUE = 10;
const int KMParameters::PREPROCESSING_MAX_GROUP_DEFAULT_VALUE = 10;
const int KMParameters::REPLICATE_NUMBER_DEFAULT_VALUE = 10;
const int KMParameters::MINI_BATCH_SIZE_DEFAULT_VALUE = 1000;
const char* KMParameters::AUTO_COMPUTED_LABEL = "Automatically computed";
const char* KMParameters::MODL_LABEL = "MODL";
const char* KMParameters::BASIC_GROUPING_LABEL = "Basic grouping + binarization";
const char* KMParameters::RANK_NORMALIZATION_LABEL = "Rank normalization";
const char* KMParameters::SOURCE_CONDITIONAL_INFO_LABEL = "Source conditional info";
const char* KMParameters::HAMMING_CONDITIONAL_INFO_CONTINUOUS_LABEL = "Hamming conditional info (continuous)";
const char* KMParameters::HAMMING_CONDITIONAL_INFO_CATEGORICAL_LABEL = "Hamming conditional info (categorical)";
const char* KMParameters::CONDITIONAL_INFO_WITH_PRIORS_CONTINUOUS_LABEL = "Conditional info with priors (continuous)";
const char* KMParameters::CONDITIONAL_INFO_WITH_PRIORS_CATEGORICAL_LABEL = "Conditional info with priors (categorical)";
const char* KMParameters::ENTROPY_CONTINUOUS_LABEL = "Entropy (continuous)";
const char* KMParameters::ENTROPY_CATEGORICAL_LABEL = "Entropy (categorical)";
const char* KMParameters::ENTROPY_WITH_PRIORS_CONTINUOUS_LABEL = "Entropy with priors (continuous)";
const char* KMParameters::ENTROPY_WITH_PRIORS_CATEGORICAL_LABEL = "Entropy with priors (categorical)";
const char* KMParameters::CENTER_REDUCTION_LABEL = "Center reduction";
const char* KMParameters::BINARIZATION_LABEL = "Binarization";
const char* KMParameters::NORMALIZATION_LABEL = "Normalization";
const char* KMParameters::UNUSED_VARIABLE_LABEL = "Unused variables";
const char* KMParameters::NO_PREPROCESSING_LABEL = "No preprocessing";
const char* KMParameters::REPLICATE_DISTANCE_LABEL = "Distance min";
const char* KMParameters::REPLICATE_EVA_LABEL = "EVA max";
const char* KMParameters::REPLICATE_ARI_BY_CLUSTERS_LABEL = "ARI max (by clusters)";
const char* KMParameters::REPLICATE_ARI_BY_CLASSES_LABEL = "ARI max (by classes)";
const char* KMParameters::REPLICATE_VARIATION_OF_INFORMATION_LABEL = "Variation of information min";
const char* KMParameters::REPLICATE_PREDICTIVE_CLUSTERING_LABEL = "Predictive clustering";
const char* KMParameters::CENTROID_REAL_INSTANCE_LABEL = "Real instance";
const char* KMParameters::CENTROID_VIRTUAL_LABEL = "Virtual centroid";
const char* KMParameters::REPLICATE_LEVA_LABEL = "LEVA max";
const char* KMParameters::REPLICATE_DAVIES_BOULDIN_LABEL = "Davies Bouldin min";
const char* KMParameters::REPLICATE_NORMALIZED_MUTUAL_INFORMATION_BY_CLUSTERS_LABEL = "NMI by clusters";
const char* KMParameters::REPLICATE_NORMALIZED_MUTUAL_INFORMATION_BY_CLASSES_LABEL = "NMI by classes";

