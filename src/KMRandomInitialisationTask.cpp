// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMRandomInitialisationTask.h"

////////////////////////////////////////////////////////////////////////////////
// Classe KMRandomInitialisationTask

#define DEBUG_TOKENS_GENERATION_TASK

KMRandomInitialisationTask::KMRandomInitialisationTask()
{
	master_parameters = NULL;
	output_centers = new PLShared_ObjectArray(new PLShared_ContinuousVector);

	DeclareTaskOutput(output_centers);
	DeclareSharedParameter(&shared_livKMeanAttributesLoadIndexes);
	DeclareSharedParameter(&shared_centersNumberToFindBySlave);
	DeclareSharedParameter(&shared_distanceType);
}

KMRandomInitialisationTask::~KMRandomInitialisationTask()
{
	delete output_centers;
	master_centers.DeleteAll();
}


boolean KMRandomInitialisationTask::FindCenters(KWDatabase* inputDatabase)
{
	return RunDatabaseTask(inputDatabase);
}


boolean KMRandomInitialisationTask::MasterInitialize()
{
	assert(master_parameters != NULL);
	boolean bTrace = false;

	// Appel a la methode ancetre
	if (not KWDatabaseTask::MasterInitialize())
		return false;

	shared_livKMeanAttributesLoadIndexes.SetLoadIndexVector(master_parameters->GetKMeanAttributesLoadIndexes().Clone());
	shared_distanceType = master_parameters->GetDistanceType();

	if (master_parameters->GetKValue() < PLParallelTask::GetProcessNumber())
		shared_centersNumberToFindBySlave = master_parameters->GetKValue();
	else
		if (master_parameters->GetKValue() == PLParallelTask::GetProcessNumber())
			shared_centersNumberToFindBySlave = 1;
		else {
			if (master_parameters->GetKValue() % PLParallelTask::GetProcessNumber() != 0)
				shared_centersNumberToFindBySlave = (master_parameters->GetKValue() / PLParallelTask::GetProcessNumber()) + 1;
			else
				shared_centersNumberToFindBySlave = (master_parameters->GetKValue() / PLParallelTask::GetProcessNumber());
		}

	if (bTrace)
		AddSimpleMessage("Slaves number : " + ALString(IntToString(PLParallelTask::GetProcessNumber())) + ALString(", max number of centers to find, by slave : ") + ALString(IntToString(shared_centersNumberToFindBySlave)));

	return true;
}

boolean KMRandomInitialisationTask::MasterAggregateResults()
{
	boolean bOk;
	int iIdenticalValues = 0;
	boolean bTrace = false;

	if (master_centers.GetSize() >= master_parameters->GetKValue()) {
		// on a deja assez de centres, ignorer les eventuels autres resultats renvoyes par les esclaves
		if (bTrace)
			AddMessage("MasterAggregateResults - no more centers needed, ignoring slave results");
		return KWDatabaseTask::MasterAggregateResults();
	}

	if (bTrace) {
		AddMessage("MasterAggregateResults - aggregating slave centers : " + ALString(IntToString(output_centers->GetObjectArray()->GetSize())) +
			", to existing master centers : " + ALString(IntToString(master_centers.GetSize())));
	}

	for (int i = 0; i < output_centers->GetObjectArray()->GetSize(); i++) {

		if (master_centers.GetSize() >= master_parameters->GetKValue())
			break; // on a deja assez de centres

		ContinuousVector* cv_slave = cast(ContinuousVector*, output_centers->GetObjectArray()->GetAt(i));

		if (IsDuplicateCenter(cv_slave, &master_centers))
			iIdenticalValues++;
		else {
			master_centers.Add(cv_slave->Clone());
		}
	}
	if (bTrace) {
		AddMessage("MasterAggregateResults has now : " + ALString(IntToString(master_centers.GetSize())));
		AddMessage("MasterAggregateResults - identical values detected : " + ALString(IntToString(iIdenticalValues)));
	}

	// Appel a la methode ancetre
	bOk = KWDatabaseTask::MasterAggregateResults();

	return bOk;
}

boolean  KMRandomInitialisationTask::SlaveProcessExploitDatabase()
{
	boolean bOk = true;
	KWDatabase* sourceDatabase;
	longint lObjectNumber;
	longint lRecordNumber;
	KWMTDatabaseMapping* mapping;
	KWObjectKey lastRootObjectKey;
	KWObject* kwoObject;
	ALString sChunkFileName;
	PLDataTableDriverTextFile* rootDriver;
	double dProgression;
	ALString sTmp;
	boolean bTrace = false;

	slave_continueCentersSearching = true;
	slave_identicalValues = 0;

	// Acces a la base source
	sourceDatabase = shared_sourceDatabase.GetDatabase();

	// Parcours des objets de la base
	lObjectNumber = 0;
	lRecordNumber = 0;
	if (bOk)
	{
		// Dans le cas multi-tables, acces au driver de la table racine, pour la gestion de la progression
		rootDriver = NULL;
		if (sourceDatabase->IsMultiTableTechnology())
		{
			mapping = cast(KWMTDatabaseMapping*, shared_sourceDatabase.GetMTDatabase()->GetMultiTableMappings()->GetAt(0));
			rootDriver = shared_sourceDatabase.GetMTDatabase()->GetDriverAt(mapping);
		}
		// Sinon, on prend le drive de la base mono-table
		else
			rootDriver = shared_sourceDatabase.GetSTDatabase()->GetDriver();

		// Parcours des objets sources
		Global::ActivateErrorFlowControl();
		while (not sourceDatabase->IsEnd())
		{
			// Suivi de la tache
			if (TaskProgression::IsRefreshNecessary())
			{
				// Avancement selon le type de base
				dProgression = rootDriver->GetReadPercentage();
				TaskProgression::DisplayProgression((int)floor(dProgression * 100));

				// Message d'avancement, uniquement dans la premiere tache (la seule ou les comptes sont corrects)
				if (GetTaskIndex() == 0)
					sourceDatabase->DisplayReadTaskProgressionLabel(lRecordNumber, lObjectNumber);
			}

			// Lecture (la gestion de l'avancement se fait dans la methode Read)
			kwoObject = sourceDatabase->Read();
			lRecordNumber++;
			if (kwoObject != NULL)
			{
				lObjectNumber++;

				// Appel de la methode exploitant l'objet
				bOk = SlaveProcessExploitDatabaseObject(kwoObject);

				// Destruction de l'objet
				delete kwoObject;
				if (not bOk)
					break;
			}
			// Arret si interruption utilisateur (deja detectee avant et ayant donc rendu un objet NULL)
			else if (TaskProgression::IsInterruptionRequested())
			{
				assert(kwoObject == NULL);
				bOk = false;
				break;
			}

			// Arret si erreur de lecture
			if (sourceDatabase->IsError())
			{
				sourceDatabase->AddError(GetTaskLabel() + " interrupted because of read errors");
				bOk = false;
				break;
			}

			// arret si on a suffisamment de centres pour cet esclave
			if (not slave_continueCentersSearching) {
				break;
			}

		}
		Global::DesactivateErrorFlowControl();
	}

	// On renvoi le nombre d'object lus
	if (bOk)
	{
		output_lReadRecords = lRecordNumber;
		output_lReadObjects = lObjectNumber;
	}

	if (bTrace) {
		AddMessage("SlaveProcessExploitDatabase - nb centers found : " + ALString(IntToString(output_centers->GetObjectArray()->GetSize())));
		AddMessage("SlaveProcessExploitDatabase - identical values detected : " + ALString(IntToString(slave_identicalValues)));
	}
	return bOk;
}


boolean KMRandomInitialisationTask::SlaveProcessExploitDatabaseObject(const KWObject* kwoCandidateCenter)
{
	require(kwoCandidateCenter != NULL);
	assert(slave_continueCentersSearching);

	Continuous cCenterSum = 0;
	Continuous c;

	KWObject* newCenter = cast(KWObject*, kwoCandidateCenter);

	// construire un ContinuousVector a partir de cette instance

	const int nbAttr = newCenter->GetClass()->GetLoadedAttributeNumber();
	require(nbAttr != 0);
	assert(shared_livKMeanAttributesLoadIndexes.GetSize() == nbAttr);

	ContinuousVector* cvNewInstance = new ContinuousVector;
	cvNewInstance->SetSize(nbAttr);
	cvNewInstance->Initialize();

	for (int i = 0; i < nbAttr; i++) {

		const KWLoadIndex loadIndex = shared_livKMeanAttributesLoadIndexes.GetAt(i);

		if (loadIndex.IsValid()) {
			// il s'agit bien d'un attribut KMeans
			c = newCenter->GetContinuousValueAt(loadIndex);
			cvNewInstance->SetAt(i, c);
			cCenterSum += c;
		}
	}

	// verifier que les valeurs kmean de cette instance ne sont pas identiques a l'un des centres deja choisis par cet esclave
	ObjectArray* results = output_centers->GetObjectArray();

	if (IsDuplicateCenter(cvNewInstance, results)) {
		delete cvNewInstance;
		slave_identicalValues++;
		return true; // on continue a lire la base pour chercher de nouveaux centres
	}

	results->Add(cvNewInstance);

	// si le nombre de centres requis pour chaque esclave a deja ete trouve par cet esclave, alors on ne traitera pas les instances suivantes de la BDD
	if (results->GetSize() >= shared_centersNumberToFindBySlave)
		slave_continueCentersSearching = false;

	return true;
}

boolean KMRandomInitialisationTask::IsDuplicateCenter(const ContinuousVector* cvNewCenter, const ObjectArray* oaExistingCenters) const {

	assert(oaExistingCenters != NULL);
	assert(cvNewCenter != NULL);
	assert(cvNewCenter->GetSize() > 0);

	boolean bIsDuplicate = false;

	if (oaExistingCenters->GetSize() == 0)
		return false;

	int iDistanceType = shared_distanceType;
	KMParameters::DistanceType distanceType = static_cast<KMParameters::DistanceType>(iDistanceType);

	// parcourir les centres deja trouves par cet esclave, et stockes sous la forme de ContinuousVector, afin de detecter un eventuel doublon
	for (int i = 0; i < oaExistingCenters->GetSize(); i++) {

		ContinuousVector* cvExistingCenter = cast(ContinuousVector*, oaExistingCenters->GetAt(i));

		const Continuous distance = KMClustering::GetDistanceBetween(*cvExistingCenter, *cvNewCenter, distanceType, *shared_livKMeanAttributesLoadIndexes.GetConstLoadIndexVector());
		if (distance == 0) {
			// le nouveau centre correspond a l'un des centres deja trouves : on l'ignore
			bIsDuplicate = true;
			break;
		}
	}

	return bIsDuplicate;
}

const ALString KMRandomInitialisationTask::GetTaskName() const
{
	return "Enneade clusters random initialization";
}

PLParallelTask* KMRandomInitialisationTask::Create() const
{
	return new KMRandomInitialisationTask;
}


void KMRandomInitialisationTask::SetParameters(const KMParameters* p)
{
	master_parameters = p;
}
