// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KMParameters.h"

#include "KMModelingSpecView.h"
#include "KWVersion.h"

class KMModelingSpecView;

////////////////////////////////////////////////////////////
// Classe KMParametersView
/// Editeur de KMParameters

class KMParametersView : public UIObjectView
{
public:

	// Constructeur
	KMParametersView();
	~KMParametersView();


	////////////////////////////////////////////////////////
	// Redefinition des methodes a reimplementer obligatoirement

	// Mise a jour de l'objet par les valeurs de l'interface
	void EventUpdate(Object* object);

	// Mise a jour des valeurs de l'interface par l'objet
	void EventRefresh(Object* object);


	// Libelles utilisateur
	const ALString GetClassLabel() const;

	//## Custom declarations

	// libelles
	static const char* K_LABEL;
	static const char* LOCAL_MODEL_TYPE_LABEL;
	static const char* LOCAL_MODEL_USE_MODL_LABEL;
	static const char* KMPARAMETERS_LABEL;
	static const char* DISTANCE_TYPE_LABEL;
	static const char* L1_NORM_LABEL;
	static const char* L2_NORM_LABEL;
	static const char* COSINUS_NORM_LABEL;
	static const char* MAX_ITERATIONS_LABEL;
	static const char* BISECTING_MAX_ITERATIONS_LABEL;
	static const char* CONTINUOUS_PREPROCESSING_LABEL;
	static const char* CATEGORICAL_PREPROCESSING_LABEL;
	static const char* CLUSTERS_CENTERS_LABEL;
	static const char* NONE_LABEL;
	static const char* NOT_USED_LABEL;
	static const char* RANDOM_LABEL;
	static const char* SAMPLE_LABEL;
	static const char* KMEAN_PLUS_PLUS_LABEL;
	static const char* ROCCHIO_SPLIT_LABEL;
	static const char* KMEAN_PLUS_PLUS_R_LABEL;
	static const char* BISECTING_LABEL;
	static const char* MIN_MAX_RANDOM_LABEL;
	static const char* MIN_MAX_DETERMINISTIC_LABEL;
	static const char* PCA_PART_LABEL;
	static const char* CLASS_DECOMPOSITION_LABEL;
	static const char* EPSILON_VALUE_LABEL;
	static const char* EPSILON_MAX_ITERATIONS_LABEL;
	static const char* CENTROID_COMPUTING_TYPE_LABEL;
	static const char* CENTROID_TYPE_LABEL;
	static const char* SIMPLIFIED_MODELING_LABEL;
	static const char* REPLICATE_NUMBER_LABEL;
	static const char* MINI_BATCH_SIZE_LABEL;
	static const char* MINI_BATCH_MODE_LABEL;
	static const char* BISECTING_REPLICATE_NUMBER_LABEL;
	static const char* REPLICATE_CHOICE_LABEL;
	static const char* REPLICATE_POST_OPTIMIZATION_LABEL;
	static const char* VNS_LEVEL_LABEL;
	static const char* REPLICATE_POST_OPTIMIZATION_FAST_LABEL;
	static const char* PREPROCESSING_MAX_INTERVAL_LABEL;
	static const char* PREPROCESSING_MAX_GROUP_LABEL;
	static const char* PREPROCESSING_SUPERVISED_MAX_INTERVAL_LABEL;
	static const char* PREPROCESSING_SUPERVISED_MAX_GROUP_LABEL;
	static const char* VERBOSE_MODE_LABEL;
	static const char* PARALLEL_MODE_LABEL;
	static const char* BISECTING_VERBOSE_MODE_LABEL;
	static const char* DETAILED_STATISTICS_LABEL;
	static const char* MAX_EVALUATED_ATTRIBUTES_NUMBER_LABEL;
	static const char* LOCAL_MODEL_NB_LABEL;
	static const char* LOCAL_MODEL_SNB_LABEL;
	static const char* KEEP_NUL_LEVEL_LABEL;

	// identifiants de champs
	static const char* K_FIELD_NAME;
	static const char* KMPARAMETERS_FIELD_NAME;
	static const char* KMPARAMETERS_KNN_FIELD_NAME;
	static const char* DISTANCE_TYPE_FIELD_NAME;
	static const char* MAX_ITERATIONS_FIELD_NAME;
	static const char* BISECTING_MAX_ITERATIONS_FIELD_NAME;
	static const char* CONTINUOUS_PREPROCESSING_FIELD_NAME;
	static const char* CATEGORICAL_PREPROCESSING_FIELD_NAME;
	static const char* CLUSTERS_CENTERS_FIELD_NAME;
	static const char* EPSILON_VALUE_FIELD_NAME;
	static const char* EPSILON_MAX_ITERATIONS_FIELD_NAME;
	static const char* CENTROID_TYPE_FIELD_NAME;
	static const char* LOCAL_MODEL_TYPE_FIELD_NAME;
	static const char* LOCAL_MODEL_USE_MODL_FIELD_NAME;
	static const char* SET_CENTROIDS_TO_NEAREST_REAL_INSTANCES_FIELD_NAME;
	static const char* REPLICATE_NUMBER_FIELD_NAME;
	static const char* MINI_BATCH_SIZE_FIELD_NAME;
	static const char* MINI_BATCH_MODE_FIELD_NAME;
	static const char* BISECTING_REPLICATE_NUMBER_FIELD_NAME;
	static const char* REPLICATE_CHOICE_FIELD_NAME;
	static const char* REPLICATE_POST_OPTIMIZATION_FIELD_NAME;
	static const char* POST_OPTIMIZATION_VNS_LEVEL_FIELD_NAME;
	static const char* PREPROCESSING_MAX_INTERVAL_FIELD_NAME;
	static const char* PREPROCESSING_MAX_GROUP_FIELD_NAME;
	static const char* PREPROCESSING_SUPERVISED_MAX_INTERVAL_FIELD_NAME;
	static const char* PREPROCESSING_SUPERVISED_MAX_GROUP_FIELD_NAME;
	static const char* VERBOSE_MODE_FIELD_NAME;
	static const char* PARALLEL_MODE_FIELD_NAME;
	static const char* BISECTING_VERBOSE_MODE_FIELD_NAME;
	static const char* DETAILED_STATISTICS_FIELD_NAME;
	static const char* MAX_EVALUATED_ATTRIBUTES_NUMBER_FIELD_NAME;
	static const char* KEEP_NUL_LEVEL_FIELD_NAME;

	//##
	////////////////////////////////////////////////////////
	//// Implementation
protected:

	const KMModelingSpecView* modelingSpecView;

};
