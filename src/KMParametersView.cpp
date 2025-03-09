// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMParametersView.h"

KMParametersView::KMParametersView()
{
	SetIdentifier(KMPARAMETERS_FIELD_NAME);
	SetLabel(KMPARAMETERS_LABEL);

	AddIntField(K_FIELD_NAME, K_LABEL, KMParameters::K_DEFAULT_VALUE);
	AddStringField(LOCAL_MODEL_TYPE_FIELD_NAME, LOCAL_MODEL_TYPE_LABEL, KMParametersView::NONE_LABEL);
	AddBooleanField(LOCAL_MODEL_USE_MODL_FIELD_NAME, LOCAL_MODEL_USE_MODL_LABEL, true);
	AddBooleanField(DETAILED_STATISTICS_FIELD_NAME, DETAILED_STATISTICS_LABEL, true);
	AddBooleanField(VERBOSE_MODE_FIELD_NAME, VERBOSE_MODE_LABEL, false);
	AddIntField(MAX_EVALUATED_ATTRIBUTES_NUMBER_FIELD_NAME, MAX_EVALUATED_ATTRIBUTES_NUMBER_LABEL, 0);
	AddIntField(REPLICATE_NUMBER_FIELD_NAME, REPLICATE_NUMBER_LABEL, KMParameters::REPLICATE_NUMBER_DEFAULT_VALUE);
	AddStringField(REPLICATE_CHOICE_FIELD_NAME, REPLICATE_CHOICE_LABEL, KMParameters::AUTO_COMPUTED_LABEL);
	AddStringField(REPLICATE_POST_OPTIMIZATION_FIELD_NAME, REPLICATE_POST_OPTIMIZATION_LABEL, KMParameters::NoOptimization);
	AddIntField(POST_OPTIMIZATION_VNS_LEVEL_FIELD_NAME, VNS_LEVEL_LABEL, 0);
	AddStringField(CONTINUOUS_PREPROCESSING_FIELD_NAME, CONTINUOUS_PREPROCESSING_LABEL, KMParameters::AUTO_COMPUTED_LABEL);
	AddStringField(CATEGORICAL_PREPROCESSING_FIELD_NAME, CATEGORICAL_PREPROCESSING_LABEL, KMParameters::AUTO_COMPUTED_LABEL);
	AddStringField(CLUSTERS_CENTERS_FIELD_NAME, CLUSTERS_CENTERS_LABEL, KMParameters::AUTO_COMPUTED_LABEL);

	AddBooleanField(MINI_BATCH_MODE_FIELD_NAME, MINI_BATCH_MODE_LABEL, false);
	AddIntField(MINI_BATCH_SIZE_FIELD_NAME, MINI_BATCH_SIZE_LABEL, KMParameters::MINI_BATCH_SIZE_DEFAULT_VALUE);
	AddIntField(MAX_ITERATIONS_FIELD_NAME, MAX_ITERATIONS_LABEL, 0);
	AddIntField(PREPROCESSING_MAX_INTERVAL_FIELD_NAME, PREPROCESSING_MAX_INTERVAL_LABEL, 0);
	AddIntField(PREPROCESSING_MAX_GROUP_FIELD_NAME, PREPROCESSING_MAX_GROUP_LABEL, 0);
	AddIntField(PREPROCESSING_SUPERVISED_MAX_INTERVAL_FIELD_NAME, PREPROCESSING_SUPERVISED_MAX_INTERVAL_LABEL, 0);
	AddIntField(PREPROCESSING_SUPERVISED_MAX_GROUP_FIELD_NAME, PREPROCESSING_SUPERVISED_MAX_GROUP_LABEL, 0);
	AddStringField(DISTANCE_TYPE_FIELD_NAME, DISTANCE_TYPE_LABEL, L2_NORM_LABEL);
	AddDoubleField(EPSILON_VALUE_FIELD_NAME, EPSILON_VALUE_LABEL, KMParameters::EPSILON_DEFAULT_VALUE);
	AddIntField(EPSILON_MAX_ITERATIONS_FIELD_NAME, EPSILON_MAX_ITERATIONS_LABEL, KMParameters::EPSILON_MAX_ITERATIONS_DEFAULT_VALUE);
	AddStringField(CENTROID_TYPE_FIELD_NAME, CENTROID_TYPE_LABEL, KMParameters::CENTROID_VIRTUAL_LABEL);
	AddBooleanField(BISECTING_VERBOSE_MODE_FIELD_NAME, BISECTING_VERBOSE_MODE_LABEL, false);
	AddIntField(BISECTING_REPLICATE_NUMBER_FIELD_NAME, BISECTING_REPLICATE_NUMBER_LABEL, KMParameters::REPLICATE_NUMBER_DEFAULT_VALUE);
	AddIntField(BISECTING_MAX_ITERATIONS_FIELD_NAME, BISECTING_MAX_ITERATIONS_LABEL, 0);
	AddBooleanField(KEEP_NUL_LEVEL_FIELD_NAME, KEEP_NUL_LEVEL_LABEL, false);
	AddBooleanField(PARALLEL_MODE_FIELD_NAME, PARALLEL_MODE_LABEL, false);

	// Parametrage des styles;
	GetFieldAt(K_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(POST_OPTIMIZATION_VNS_LEVEL_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(REPLICATE_NUMBER_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(MINI_BATCH_SIZE_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(LOCAL_MODEL_TYPE_FIELD_NAME)->SetStyle("ComboBox");
	GetFieldAt(REPLICATE_CHOICE_FIELD_NAME)->SetStyle("ComboBox");
	GetFieldAt(CONTINUOUS_PREPROCESSING_FIELD_NAME)->SetStyle("ComboBox");
	GetFieldAt(CATEGORICAL_PREPROCESSING_FIELD_NAME)->SetStyle("ComboBox");
	GetFieldAt(DISTANCE_TYPE_FIELD_NAME)->SetStyle("ComboBox");
	GetFieldAt(REPLICATE_POST_OPTIMIZATION_FIELD_NAME)->SetStyle("ComboBox");
	GetFieldAt(PREPROCESSING_SUPERVISED_MAX_INTERVAL_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(PREPROCESSING_SUPERVISED_MAX_GROUP_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(CENTROID_TYPE_FIELD_NAME)->SetStyle("ComboBox");
	GetFieldAt(MAX_ITERATIONS_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(BISECTING_MAX_ITERATIONS_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(BISECTING_REPLICATE_NUMBER_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(CLUSTERS_CENTERS_FIELD_NAME)->SetStyle("ComboBox");
	GetFieldAt(EPSILON_MAX_ITERATIONS_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(PREPROCESSING_MAX_INTERVAL_FIELD_NAME)->SetStyle("Spinner");
	GetFieldAt(PREPROCESSING_MAX_GROUP_FIELD_NAME)->SetStyle("Spinner");

	// Parametrage des libelles
	GetFieldAt(CONTINUOUS_PREPROCESSING_FIELD_NAME)->SetParameters(
		ALString(KMParameters::AUTO_COMPUTED_LABEL) + "\n" +
		ALString(KMParameters::NO_PREPROCESSING_LABEL) + "\n" +
		ALString(KMParameters::UNUSED_VARIABLE_LABEL) + "\n" +
		ALString(KMParameters::RANK_NORMALIZATION_LABEL) + "\n" +
		ALString(KMParameters::CENTER_REDUCTION_LABEL) + "\n" +
		ALString(KMParameters::BINARIZATION_LABEL) + "\n" +
		ALString(KMParameters::HAMMING_CONDITIONAL_INFO_CONTINUOUS_LABEL) + "\n" +
		ALString(KMParameters::CONDITIONAL_INFO_WITH_PRIORS_CONTINUOUS_LABEL) + "\n" +
		ALString(KMParameters::ENTROPY_CONTINUOUS_LABEL) + "\n" +
		ALString(KMParameters::ENTROPY_WITH_PRIORS_CONTINUOUS_LABEL) + "\n" +
		ALString(KMParameters::NORMALIZATION_LABEL)
	);
	GetFieldAt(CATEGORICAL_PREPROCESSING_FIELD_NAME)->SetParameters(
		ALString(KMParameters::AUTO_COMPUTED_LABEL) + "\n" +
		ALString(KMParameters::UNUSED_VARIABLE_LABEL) + "\n" +
		ALString(KMParameters::BINARIZATION_LABEL) + "\n" +
		ALString(KMParameters::HAMMING_CONDITIONAL_INFO_CATEGORICAL_LABEL) + "\n" +
		ALString(KMParameters::CONDITIONAL_INFO_WITH_PRIORS_CATEGORICAL_LABEL) + "\n" +
		ALString(KMParameters::ENTROPY_CATEGORICAL_LABEL) + "\n" +
		ALString(KMParameters::ENTROPY_WITH_PRIORS_CATEGORICAL_LABEL) + "\n" +
		ALString(KMParameters::BASIC_GROUPING_LABEL)
	);

	if (GetLearningExpertMode())
		GetFieldAt(REPLICATE_CHOICE_FIELD_NAME)->SetParameters(
			ALString(KMParameters::AUTO_COMPUTED_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_DISTANCE_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_ARI_BY_CLUSTERS_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_ARI_BY_CLASSES_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_EVA_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_LEVA_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_VARIATION_OF_INFORMATION_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_PREDICTIVE_CLUSTERING_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_DAVIES_BOULDIN_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_NORMALIZED_MUTUAL_INFORMATION_BY_CLUSTERS_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_NORMALIZED_MUTUAL_INFORMATION_BY_CLASSES_LABEL)
		);
	else
		GetFieldAt(REPLICATE_CHOICE_FIELD_NAME)->SetParameters(
			ALString(KMParameters::AUTO_COMPUTED_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_DISTANCE_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_ARI_BY_CLUSTERS_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_PREDICTIVE_CLUSTERING_LABEL) + "\n" +
			ALString(KMParameters::REPLICATE_DAVIES_BOULDIN_LABEL)
		);

	GetFieldAt(DISTANCE_TYPE_FIELD_NAME)->SetParameters(
		ALString(L2_NORM_LABEL) + "\n" +
		ALString(L1_NORM_LABEL) + "\n" +
		ALString(COSINUS_NORM_LABEL)
	);
	GetFieldAt(CENTROID_TYPE_FIELD_NAME)->SetParameters(
		ALString(KMParameters::CENTROID_VIRTUAL_LABEL) + "\n" +
		ALString(KMParameters::CENTROID_REAL_INSTANCE_LABEL)
	);
	GetFieldAt(CLUSTERS_CENTERS_FIELD_NAME)->SetParameters(
		ALString(KMParameters::AUTO_COMPUTED_LABEL) + "\n" +
		ALString(RANDOM_LABEL) + "\n" +
		ALString(SAMPLE_LABEL) + "\n" +
		ALString(KMEAN_PLUS_PLUS_LABEL) + "\n" +
		ALString(KMEAN_PLUS_PLUS_R_LABEL) + "\n" +
		ALString(ROCCHIO_SPLIT_LABEL) + "\n" +
		ALString(MIN_MAX_RANDOM_LABEL) + "\n" +
		ALString(MIN_MAX_DETERMINISTIC_LABEL) + "\n" +
		ALString(PCA_PART_LABEL) + "\n" +
		ALString(CLASS_DECOMPOSITION_LABEL) + "\n" +
		ALString(BISECTING_LABEL)
	);
	GetFieldAt(LOCAL_MODEL_TYPE_FIELD_NAME)->SetParameters(
		ALString(NONE_LABEL) + "\n" +
		ALString(LOCAL_MODEL_NB_LABEL) + "\n" +
		ALString(LOCAL_MODEL_SNB_LABEL)
	);

	GetFieldAt(REPLICATE_POST_OPTIMIZATION_FIELD_NAME)->SetParameters(
		ALString(REPLICATE_POST_OPTIMIZATION_FAST_LABEL) + "\n" +
		ALString(NONE_LABEL)
	);

	// Parametrage des plages de valeurs
	cast(UIIntElement*, GetFieldAt(MAX_EVALUATED_ATTRIBUTES_NUMBER_FIELD_NAME))->SetMinValue(0);
	cast(UIIntElement*, GetFieldAt(MAX_EVALUATED_ATTRIBUTES_NUMBER_FIELD_NAME))->SetMaxValue(1000000);

	cast(UIIntElement*, GetFieldAt(K_FIELD_NAME))->SetMinValue(1);
	cast(UIIntElement*, GetFieldAt(K_FIELD_NAME))->SetMaxValue(KMParameters::K_MAX_VALUE);

	cast(UIIntElement*, GetFieldAt(REPLICATE_NUMBER_FIELD_NAME))->SetMinValue(1);
	cast(UIIntElement*, GetFieldAt(REPLICATE_NUMBER_FIELD_NAME))->SetMaxValue(KMParameters::REPLICATE_NUMBER_MAX_VALUE);

	cast(UIIntElement*, GetFieldAt(MINI_BATCH_SIZE_FIELD_NAME))->SetMinValue(10);
	cast(UIIntElement*, GetFieldAt(MINI_BATCH_SIZE_FIELD_NAME))->SetMaxValue(KMParameters::MINI_BATCH_SIZE_MAX_VALUE);

	cast(UIIntElement*, GetFieldAt(MAX_ITERATIONS_FIELD_NAME))->SetMinValue(-1);
	cast(UIIntElement*, GetFieldAt(MAX_ITERATIONS_FIELD_NAME))->SetMaxValue(KMParameters::MAX_ITERATIONS);

	cast(UIIntElement*, GetFieldAt(POST_OPTIMIZATION_VNS_LEVEL_FIELD_NAME))->SetMinValue(0);

	cast(UIDoubleElement*, GetFieldAt(EPSILON_VALUE_FIELD_NAME))->SetMinValue(0);

	cast(UIIntElement*, GetFieldAt(EPSILON_MAX_ITERATIONS_FIELD_NAME))->SetMinValue(0);
	cast(UIIntElement*, GetFieldAt(EPSILON_MAX_ITERATIONS_FIELD_NAME))->SetMaxValue(KMParameters::EPSILON_MAX_ITERATIONS);

	cast(UIIntElement*, GetFieldAt(PREPROCESSING_MAX_INTERVAL_FIELD_NAME))->SetMinValue(0);
	cast(UIIntElement*, GetFieldAt(PREPROCESSING_MAX_GROUP_FIELD_NAME))->SetMinValue(0);

	cast(UIIntElement*, GetFieldAt(BISECTING_MAX_ITERATIONS_FIELD_NAME))->SetMinValue(-1);
	cast(UIIntElement*, GetFieldAt(BISECTING_MAX_ITERATIONS_FIELD_NAME))->SetMaxValue(KMParameters::MAX_ITERATIONS);

	cast(UIIntElement*, GetFieldAt(BISECTING_REPLICATE_NUMBER_FIELD_NAME))->SetMinValue(1);
	cast(UIIntElement*, GetFieldAt(BISECTING_REPLICATE_NUMBER_FIELD_NAME))->SetMaxValue(KMParameters::REPLICATE_NUMBER_MAX_VALUE);

	cast(UIIntElement*, GetFieldAt(PREPROCESSING_SUPERVISED_MAX_INTERVAL_FIELD_NAME))->SetMinValue(0);
	cast(UIIntElement*, GetFieldAt(PREPROCESSING_SUPERVISED_MAX_GROUP_FIELD_NAME))->SetMinValue(0);

	// Info-bulles
	GetFieldAt(MAX_EVALUATED_ATTRIBUTES_NUMBER_FIELD_NAME)->SetHelpText("Max number of variables originating from the univariate data preparation (discretizations and value groupings),"
		"\n to use as input variables in the predictor."
		"\n The evaluated variables are the ones having the highest univariate predictive importance."
		"\n This parameter allows to simplify and speed up the training phase"
		"\n (default: 0, means that all the variables are evaluated).");
	GetFieldAt(K_FIELD_NAME)->SetHelpText("Desired number of clusters.\n"
		"Depending on the initialization method and the convergence process,\n"
		"you may obtain a lower number of clusters.");
	GetFieldAt(REPLICATE_NUMBER_FIELD_NAME)->SetHelpText("A replicate is composed of a centroids initialization method, and a convergence process.");
	GetFieldAt(REPLICATE_POST_OPTIMIZATION_FIELD_NAME)->SetHelpText("Post-optimize the replicate result, by removing clusters if the \nremoving produces a better EVA (supervised mode only)");
	GetFieldAt(REPLICATE_CHOICE_FIELD_NAME)->SetHelpText("Clustering quality criterion");
	GetFieldAt(CONTINUOUS_PREPROCESSING_FIELD_NAME)->SetHelpText("Preprocessing method for continuous attributes");
	GetFieldAt(CATEGORICAL_PREPROCESSING_FIELD_NAME)->SetHelpText("Preprocessing method for categorical attributes");
	GetFieldAt(DISTANCE_TYPE_FIELD_NAME)->SetHelpText("Norm to use when computing distances between instances and/or centroids");
	GetFieldAt(DETAILED_STATISTICS_FIELD_NAME)->SetHelpText("If activated, several detailed statistics will be computed using intervals"
		"\n and modalities, and will be written in the evaluation report");
	GetFieldAt(MAX_EVALUATED_ATTRIBUTES_NUMBER_FIELD_NAME)->SetHelpText("0 is no max. If a value is set, then only the most significant "
		"\nvariables will be evaluated, based on their 'level'.");
	GetFieldAt(PREPROCESSING_SUPERVISED_MAX_INTERVAL_FIELD_NAME)->SetHelpText("Continuous processing : 'force' the maximum number of intervals"
		"\nunder its optimum level (supervised mode only)");
	GetFieldAt(PREPROCESSING_SUPERVISED_MAX_GROUP_FIELD_NAME)->SetHelpText("Categorical preprocessing : 'force' the maximum number of groups "
		"\nunder its optimum level (supervised mode only)");

	// Le parametrage expert n'est visible qu'en mode expert
	GetFieldAt(MAX_ITERATIONS_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(EPSILON_VALUE_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(EPSILON_MAX_ITERATIONS_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(CENTROID_TYPE_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(BISECTING_VERBOSE_MODE_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(BISECTING_REPLICATE_NUMBER_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(MINI_BATCH_SIZE_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(BISECTING_MAX_ITERATIONS_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(PREPROCESSING_MAX_INTERVAL_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(PREPROCESSING_MAX_GROUP_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(PREPROCESSING_SUPERVISED_MAX_INTERVAL_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(PREPROCESSING_SUPERVISED_MAX_GROUP_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(MINI_BATCH_MODE_FIELD_NAME)->SetVisible(GetLearningExpertMode());
	GetFieldAt(PARALLEL_MODE_FIELD_NAME)->SetVisible(GetLearningExpertMode());
}


KMParametersView::~KMParametersView()
{
}


void KMParametersView::EventUpdate(Object* object)
{
	KMParameters* editedObject;

	require(object != NULL);

	editedObject = cast(KMParameters*, object);

	editedObject->SetKValue(GetIntValueAt(K_FIELD_NAME));
	editedObject->SetLocalModelType(GetStringValueAt(LOCAL_MODEL_TYPE_FIELD_NAME));
	editedObject->SetLearningNumberOfReplicates(GetIntValueAt(REPLICATE_NUMBER_FIELD_NAME));
	editedObject->SetMiniBatchSize(GetIntValueAt(MINI_BATCH_SIZE_FIELD_NAME));
	editedObject->SetMiniBatchMode(GetBooleanValueAt(MINI_BATCH_MODE_FIELD_NAME));
	editedObject->SetPostOptimizationVnsLevel(GetIntValueAt(POST_OPTIMIZATION_VNS_LEVEL_FIELD_NAME));
	editedObject->SetMaxIterations(GetIntValueAt(MAX_ITERATIONS_FIELD_NAME));
	editedObject->SetBisectingMaxIterations(GetIntValueAt(BISECTING_MAX_ITERATIONS_FIELD_NAME));
	editedObject->SetPreprocessingMaxIntervalNumber(GetIntValueAt(PREPROCESSING_MAX_INTERVAL_FIELD_NAME));
	editedObject->SetPreprocessingMaxGroupNumber(GetIntValueAt(PREPROCESSING_MAX_GROUP_FIELD_NAME));
	editedObject->SetBisectingVerboseMode(GetBooleanValueAt(BISECTING_VERBOSE_MODE_FIELD_NAME));
	editedObject->SetBisectingNumberOfReplicates(GetIntValueAt(BISECTING_REPLICATE_NUMBER_FIELD_NAME));
	editedObject->SetMaxEvaluatedAttributesNumber(GetIntValueAt(MAX_EVALUATED_ATTRIBUTES_NUMBER_FIELD_NAME));
	editedObject->SetWriteDetailedStatistics(GetBooleanValueAt(DETAILED_STATISTICS_FIELD_NAME));
	editedObject->SetPreprocessingSupervisedMaxIntervalNumber(GetIntValueAt(PREPROCESSING_SUPERVISED_MAX_INTERVAL_FIELD_NAME));
	editedObject->SetPreprocessingSupervisedMaxGroupNumber(GetIntValueAt(PREPROCESSING_SUPERVISED_MAX_GROUP_FIELD_NAME));
	editedObject->SetEpsilonValue(GetDoubleValueAt(EPSILON_VALUE_FIELD_NAME));
	editedObject->SetEpsilonMaxIterations(GetIntValueAt(EPSILON_MAX_ITERATIONS_FIELD_NAME));
	editedObject->SetVerboseMode(GetBooleanValueAt(VERBOSE_MODE_FIELD_NAME));
	editedObject->SetParallelMode(GetBooleanValueAt(PARALLEL_MODE_FIELD_NAME));
	editedObject->SetReplicatePostOptimization(GetStringValueAt(REPLICATE_POST_OPTIMIZATION_FIELD_NAME));
	editedObject->SetReplicateChoice(GetStringValueAt(REPLICATE_CHOICE_FIELD_NAME));
	editedObject->SetClustersCentersInitializationMethod(GetStringValueAt(CLUSTERS_CENTERS_FIELD_NAME));
	editedObject->SetCentroidType(GetStringValueAt(CENTROID_TYPE_FIELD_NAME));
	editedObject->SetDistanceType(GetStringValueAt(DISTANCE_TYPE_FIELD_NAME));
	editedObject->SetCategoricalPreprocessingType(GetStringValueAt(CATEGORICAL_PREPROCESSING_FIELD_NAME));
	editedObject->SetContinuousPreprocessingType(GetStringValueAt(CONTINUOUS_PREPROCESSING_FIELD_NAME));
	editedObject->SetKeepNulLevelVariables(GetBooleanValueAt(KEEP_NUL_LEVEL_FIELD_NAME));
}


void KMParametersView::EventRefresh(Object* object)
{
	KMParameters* editedObject;

	require(object != NULL);

	editedObject = cast(KMParameters*, object);

	SetIntValueAt(K_FIELD_NAME, editedObject->GetKValue());
	SetStringValueAt(LOCAL_MODEL_TYPE_FIELD_NAME, editedObject->GetLocalModelTypeLabel());
	SetIntValueAt(REPLICATE_NUMBER_FIELD_NAME, editedObject->GetLearningNumberOfReplicates());
	SetIntValueAt(MINI_BATCH_SIZE_FIELD_NAME, editedObject->GetMiniBatchSize());
	SetIntValueAt(POST_OPTIMIZATION_VNS_LEVEL_FIELD_NAME, editedObject->GetPostOptimizationVnsLevel());
	SetBooleanValueAt(MINI_BATCH_MODE_FIELD_NAME, editedObject->GetMiniBatchMode());
	SetStringValueAt(REPLICATE_CHOICE_FIELD_NAME, editedObject->GetReplicateChoiceLabel());
	SetStringValueAt(DISTANCE_TYPE_FIELD_NAME, editedObject->GetDistanceTypeLabel());
	SetStringValueAt(CATEGORICAL_PREPROCESSING_FIELD_NAME, editedObject->GetCategoricalPreprocessingTypeLabel());
	SetStringValueAt(CONTINUOUS_PREPROCESSING_FIELD_NAME, editedObject->GetContinuousPreprocessingTypeLabel());
	SetBooleanValueAt(VERBOSE_MODE_FIELD_NAME, editedObject->GetVerboseMode());
	SetBooleanValueAt(PARALLEL_MODE_FIELD_NAME, editedObject->GetParallelMode());
	SetStringValueAt(REPLICATE_POST_OPTIMIZATION_FIELD_NAME, editedObject->GetReplicatePostOptimizationLabel());
	SetBooleanValueAt(DETAILED_STATISTICS_FIELD_NAME, editedObject->GetWriteDetailedStatistics());
	SetIntValueAt(PREPROCESSING_SUPERVISED_MAX_INTERVAL_FIELD_NAME, editedObject->GetPreprocessingSupervisedMaxIntervalNumber());
	SetIntValueAt(PREPROCESSING_SUPERVISED_MAX_GROUP_FIELD_NAME, editedObject->GetPreprocessingSupervisedMaxGroupNumber());
	SetIntValueAt(MAX_EVALUATED_ATTRIBUTES_NUMBER_FIELD_NAME, editedObject->GetMaxEvaluatedAttributesNumber());
	SetIntValueAt(MAX_ITERATIONS_FIELD_NAME, editedObject->GetMaxIterations());
	SetIntValueAt(BISECTING_MAX_ITERATIONS_FIELD_NAME, editedObject->GetBisectingMaxIterations());
	SetBooleanValueAt(BISECTING_VERBOSE_MODE_FIELD_NAME, editedObject->GetBisectingVerboseMode());
	SetIntValueAt(PREPROCESSING_MAX_INTERVAL_FIELD_NAME, editedObject->GetPreprocessingMaxIntervalNumber());
	SetIntValueAt(PREPROCESSING_MAX_GROUP_FIELD_NAME, editedObject->GetPreprocessingMaxGroupNumber());
	SetStringValueAt(CLUSTERS_CENTERS_FIELD_NAME, editedObject->GetClustersCentersInitializationMethodLabel());
	SetDoubleValueAt(EPSILON_VALUE_FIELD_NAME, editedObject->GetEpsilonValue());
	SetIntValueAt(EPSILON_MAX_ITERATIONS_FIELD_NAME, editedObject->GetEpsilonMaxIterations());
	SetStringValueAt(CENTROID_TYPE_FIELD_NAME, editedObject->GetCentroidTypeLabel());
	SetIntValueAt(BISECTING_REPLICATE_NUMBER_FIELD_NAME, editedObject->GetBisectingNumberOfReplicates());
	SetBooleanValueAt(KEEP_NUL_LEVEL_FIELD_NAME, editedObject->GetKeepNulLevelVariables());
}


const ALString KMParametersView::GetClassLabel() const
{
	return "Clustering parameters";
}

// libelles
const char* KMParametersView::K_LABEL = "Clusters number (K)";
const char* KMParametersView::LOCAL_MODEL_TYPE_LABEL = "Local models";
const char* KMParametersView::LOCAL_MODEL_USE_MODL_LABEL = "Always use MODL for preprocessing in local models";
const char* KMParametersView::KMPARAMETERS_LABEL = "Clustering parameters";
const char* KMParametersView::DISTANCE_TYPE_LABEL = "Distance type";
const char* KMParametersView::L1_NORM_LABEL = "L1 norm";
const char* KMParametersView::L2_NORM_LABEL = "L2 norm";
const char* KMParametersView::COSINUS_NORM_LABEL = "Cosine norm";
const char* KMParametersView::MAX_ITERATIONS_LABEL = "Iterations max number(0 = no max, -1 = no iteration)";
const char* KMParametersView::BISECTING_MAX_ITERATIONS_LABEL = "Bisecting/class decomposition iterations max number(0 = no max, -1 = no iteration)";
const char* KMParametersView::CONTINUOUS_PREPROCESSING_LABEL = "Continuous preprocessing type";
const char* KMParametersView::CATEGORICAL_PREPROCESSING_LABEL = "Categorical preprocessing type";
const char* KMParametersView::CLUSTERS_CENTERS_LABEL = "Clusters centers initialization";
const char* KMParametersView::NONE_LABEL = "None";
const char* KMParametersView::NOT_USED_LABEL = "Not used";
const char* KMParametersView::RANDOM_LABEL = "Random";
const char* KMParametersView::SAMPLE_LABEL = "Sample";
const char* KMParametersView::KMEAN_PLUS_PLUS_LABEL = "KMean++";
const char* KMParametersView::KMEAN_PLUS_PLUS_R_LABEL = "KMean++R";
const char* KMParametersView::ROCCHIO_SPLIT_LABEL = "Rocchio, then split";
const char* KMParametersView::BISECTING_LABEL = "Bisecting";
const char* KMParametersView::MIN_MAX_RANDOM_LABEL = "Min-Max (random)";
const char* KMParametersView::MIN_MAX_DETERMINISTIC_LABEL = "Min-Max (deterministic)";
const char* KMParametersView::PCA_PART_LABEL = "Variance partitioning";
const char* KMParametersView::CLASS_DECOMPOSITION_LABEL = "Class decomposition";
const char* KMParametersView::EPSILON_VALUE_LABEL = "Epsilon value";
const char* KMParametersView::EPSILON_MAX_ITERATIONS_LABEL = "Max iterations under epsilon";
const char* KMParametersView::CENTROID_COMPUTING_TYPE_LABEL = "Centroid computing type";
const char* KMParametersView::CENTROID_TYPE_LABEL = "Centroid type";
const char* KMParametersView::SIMPLIFIED_MODELING_LABEL = "Simplified modeling (supervised mode only)";
const char* KMParametersView::REPLICATE_NUMBER_LABEL = "Learning number of replicates";
const char* KMParametersView::MINI_BATCH_SIZE_LABEL = "Mini-batches size (number of instances)";
const char* KMParametersView::MINI_BATCH_MODE_LABEL = "Force mini-batch mode";
const char* KMParametersView::BISECTING_REPLICATE_NUMBER_LABEL = "Bisecting/class decomposition number of replicates";
const char* KMParametersView::REPLICATE_CHOICE_LABEL = "Best replicate selection";
const char* KMParametersView::PREPROCESSING_MAX_INTERVAL_LABEL = "Unsupervised mode: max intervals number (0 = no max)";
const char* KMParametersView::PREPROCESSING_MAX_GROUP_LABEL = "Unsupervised mode: max groups number (0 = no max)";
const char* KMParametersView::PREPROCESSING_SUPERVISED_MAX_INTERVAL_LABEL = "Supervised mode: max intervals number (0 = no max)";
const char* KMParametersView::PREPROCESSING_SUPERVISED_MAX_GROUP_LABEL = "Supervised mode: max groups number (0 = no max)";
const char* KMParametersView::VERBOSE_MODE_LABEL = "Verbose mode";
const char* KMParametersView::PARALLEL_MODE_LABEL = "Parallel mode";
const char* KMParametersView::BISECTING_VERBOSE_MODE_LABEL = "Bisecting/class decomposition verbose mode";
const char* KMParametersView::DETAILED_STATISTICS_LABEL = "Write detailed statistics in reports";
const char* KMParametersView::MAX_EVALUATED_ATTRIBUTES_NUMBER_LABEL = "Max number of used variables (supervised mode only, 0 = no max)";
const char* KMParametersView::LOCAL_MODEL_SNB_LABEL = "Selective Naive Bayes";
const char* KMParametersView::LOCAL_MODEL_NB_LABEL = "Naive Bayes";
const char* KMParametersView::REPLICATE_POST_OPTIMIZATION_LABEL = "Best replicate post-optimization";
const char* KMParametersView::VNS_LEVEL_LABEL = "Post-optimization VNS level (0 = no VNS)";
const char* KMParametersView::REPLICATE_POST_OPTIMIZATION_FAST_LABEL = "Fast post-optimization";
const char* KMParametersView::KEEP_NUL_LEVEL_LABEL = "Keep all variables in case of unsupervised preprocessing (supervised mode only)";

// identifiants de champs
const char* KMParametersView::K_FIELD_NAME = "K";
const char* KMParametersView::KMPARAMETERS_FIELD_NAME = "KMParameters";
const char* KMParametersView::KMPARAMETERS_KNN_FIELD_NAME = "KMParametersKNN";
const char* KMParametersView::DISTANCE_TYPE_FIELD_NAME = "DistanceType";
const char* KMParametersView::MAX_ITERATIONS_FIELD_NAME = "MaxIterations";
const char* KMParametersView::BISECTING_MAX_ITERATIONS_FIELD_NAME = "BisectingMaxIterations";
const char* KMParametersView::CONTINUOUS_PREPROCESSING_FIELD_NAME = "ContinuousPreprocessingType";
const char* KMParametersView::CATEGORICAL_PREPROCESSING_FIELD_NAME = "CategoricalPreprocessingType";
const char* KMParametersView::CLUSTERS_CENTERS_FIELD_NAME = "ClustersCentersInitialization";
const char* KMParametersView::EPSILON_VALUE_FIELD_NAME = "EpsilonValue";
const char* KMParametersView::EPSILON_MAX_ITERATIONS_FIELD_NAME = "EpsilonMaxIterations";
const char* KMParametersView::CENTROID_TYPE_FIELD_NAME = "CentroidType";
const char* KMParametersView::LOCAL_MODEL_TYPE_FIELD_NAME = "LocalModelType";
const char* KMParametersView::LOCAL_MODEL_USE_MODL_FIELD_NAME = "LocalModelUseMODL";
const char* KMParametersView::REPLICATE_NUMBER_FIELD_NAME = "NumberOfReplicates";
const char* KMParametersView::MINI_BATCH_SIZE_FIELD_NAME = "MiniBatchSize";
const char* KMParametersView::MINI_BATCH_MODE_FIELD_NAME = "MiniBatchMode";
const char* KMParametersView::BISECTING_REPLICATE_NUMBER_FIELD_NAME = "BisectingNumberOfReplicates";
const char* KMParametersView::REPLICATE_CHOICE_FIELD_NAME = "ReplicateChoice";
const char* KMParametersView::REPLICATE_POST_OPTIMIZATION_FIELD_NAME = "ReplicatePostOptimization";
const char* KMParametersView::POST_OPTIMIZATION_VNS_LEVEL_FIELD_NAME = "PostOptimizationVnsLevel";
const char* KMParametersView::PREPROCESSING_MAX_INTERVAL_FIELD_NAME = "p";
const char* KMParametersView::PREPROCESSING_MAX_GROUP_FIELD_NAME = "q";
const char* KMParametersView::PREPROCESSING_SUPERVISED_MAX_INTERVAL_FIELD_NAME = "SupervisedMaxInterval";
const char* KMParametersView::PREPROCESSING_SUPERVISED_MAX_GROUP_FIELD_NAME = "SupervisedMaxGroup";
const char* KMParametersView::VERBOSE_MODE_FIELD_NAME = "VerboseMode";
const char* KMParametersView::PARALLEL_MODE_FIELD_NAME = "ParallelMode";
const char* KMParametersView::BISECTING_VERBOSE_MODE_FIELD_NAME = "BisectingVerboseMode";
const char* KMParametersView::DETAILED_STATISTICS_FIELD_NAME = "WriteDetailedStatistics";
const char* KMParametersView::MAX_EVALUATED_ATTRIBUTES_NUMBER_FIELD_NAME = "MaxEvaluatedAttributesNumber";
const char* KMParametersView::KEEP_NUL_LEVEL_FIELD_NAME = "KeepNulLevel";

