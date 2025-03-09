// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWDatabaseTask.h"
#include "KMLearningProblem.h"

class KMLearningProblem;

////////////////////////////////////////////////////////////////////////////////
/// Tache parallelisee d'initialisation de clustering, selon l'algo Random

class KMRandomInitialisationTask : public KWDatabaseTask
{
public:
	// Constructeur
	KMRandomInitialisationTask();
	~KMRandomInitialisationTask();

	/** recherche des centres, a partir d'une database */
	boolean FindCenters(KWDatabase* inputDatabase);

	/** parametrage general kmeans */
	void SetParameters(const KMParameters*);

	/** liste des centres trouves */
	const ObjectArray& GetCenters() const;

	//////////////////////////////////////////////////////////////////////////////
	///// Implementation
protected:

	// Reimplementation des etapes du DatabaseTask (methodes virtuelles)
	const ALString GetTaskName() const override;
	PLParallelTask* Create() const override;
	boolean MasterInitialize() override;
	boolean MasterAggregateResults() override;
	boolean SlaveProcessExploitDatabase() override;
	boolean SlaveProcessExploitDatabaseObject(const KWObject* kwoObject) override;
	boolean IsDuplicateCenter(const ContinuousVector* newCenter, const ObjectArray* existingCenters) const;

	// variables membres du maitre
	const KMParameters* master_parameters;
	ObjectArray master_centers;// liste des centres trouvés (liste de ContinuousVector *)

	// variables membres des esclaves
	/** indique si un esclave doit continuer a parcourir sa portion de BDD afin de trouver de nouveaux centres */
	boolean slave_continueCentersSearching;

	/** nombre de fois ou une instance candidate pour devenir un centre a ete rejetee, car elle avait des valeurs identiques a celles d'un centre deja choisi */
	int slave_identicalValues;

	// variables partagees
	PLShared_LoadIndexVector shared_livKMeanAttributesLoadIndexes;
	PLShared_Int shared_centersNumberToFindBySlave;
	PLShared_ObjectArray* output_centers;
	PLShared_Int shared_distanceType;
};

inline const ObjectArray& KMRandomInitialisationTask::GetCenters() const {
	return master_centers;
}

