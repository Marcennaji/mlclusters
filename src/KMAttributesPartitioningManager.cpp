// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMAttributesPartitioningManager.h"
#include "KMClustering.h"

KMAttributesPartitioningManager::KMAttributesPartitioningManager()
{
}


KMAttributesPartitioningManager::~KMAttributesPartitioningManager(void)
{
	POSITION position = odAttributesPartitions.GetStartPosition();
	ALString key;
	Object* oCurrent;

	while (position != NULL) {

		odAttributesPartitions.GetNextAssoc(position, key, oCurrent);

		ObjectArray* oaModalities = cast(ObjectArray*, oCurrent);
		oaModalities->DeleteAll();
		delete oaModalities;
	}

	position = odAtomicModalities.GetStartPosition();

	while (position != NULL) {

		odAtomicModalities.GetNextAssoc(position, key, oCurrent);

		ObjectArray* oaModalities = cast(ObjectArray*, oCurrent);
		oaModalities->DeleteAll();
		delete oaModalities;
	}
}

void KMAttributesPartitioningManager::AddAttributePartitions(const ALString attributeName, ObjectArray* partitions) {

	assert(partitions != NULL);
	assert(attributeName != "");

	// supprimer d'abord les anciennes valeurs correspondant a cet attribut
	Object* o = odAttributesPartitions.Lookup(attributeName);
	if (o != NULL) {
		ObjectArray* oaModalities = cast(ObjectArray*, o);
		oaModalities->DeleteAll();
		delete oaModalities;
	}
	odAttributesPartitions.SetAt(attributeName, partitions);
}
void KMAttributesPartitioningManager::AddAtomicModalities(const ALString attributeName, ObjectArray* modalities) {

	assert(modalities != NULL);
	assert(attributeName != "");

	// supprimer d'abord les anciennes valeurs correspondant a cet attribut
	Object* o = odAtomicModalities.Lookup(attributeName);
	if (o != NULL) {
		ObjectArray* oaModalities = cast(ObjectArray*, o);
		oaModalities->DeleteAll();
		delete oaModalities;
	}

	odAtomicModalities.SetAt(attributeName, modalities);

}

const ObjectDictionary& KMAttributesPartitioningManager::GetPartitions() const {
	return odAttributesPartitions;
}

const ObjectDictionary& KMAttributesPartitioningManager::GetAtomicModalities() const {
	return odAtomicModalities;
}

void KMAttributesPartitioningManager::AddValueGroups(KWDRValueGroups* kwdrGroups, const ALString attributeName, const int maxValuesToAdd, const bool supervisedMode) {

	ObjectArray partitions;

	// elaborer les libelles de modalités groupées
	for (int i = 0; i < kwdrGroups->GetPartNumber(); i++) {

		StringObject* s = new StringObject();

		KWDRValueGroup* valueGroup = cast(KWDRValueGroup*, kwdrGroups->GetOperandAt(i)->GetDerivationRule());

		// Fabrication du libelle a partir des valeurs du groupes
		int nNumber = 0;
		ALString sLabel = "{";
		for (int nValue = 0; nValue < valueGroup->GetValueNumber(); nValue++)
		{
			ALString sValue = valueGroup->GetValueAt(nValue).GetValue();

			// fabriquer le libelle en limitant du nombre de valeurs stockees
			if (nNumber < maxValuesToAdd or sValue == Symbol::GetStarValue())
			{
				if (nNumber > 0)
					sLabel += ", ";
				sLabel += sValue;

			}
			else
			{
				if (nNumber == maxValuesToAdd)
					sLabel += ", ...";
			}
			nNumber++;

		}
		sLabel += "}";

		s->SetString(sLabel);

		partitions.Add(s);
	}

	AddAttributePartitions(attributeName, partitions.Clone());

	// stocker les libelles de modalités NON groupées, s'il n'y a pas plus de 10 modalités en tout (sans compter la modalite '*') :

	int nbModalities = 0;
	for (int i = 0; i < kwdrGroups->GetValueGroupNumber(); i++) {
		KWDRValueGroup* valueGroup = kwdrGroups->GetValueGroupAt(i);
		nbModalities += valueGroup->GetValueNumber();
	}

	if (nbModalities > 11) // ne pas compter la modalite '*', toujours presente dans l'un des groupes
		return;

	ObjectArray atomicModalities;

	for (int i = 0; i < kwdrGroups->GetPartNumber(); i++) {

		KWDRValueGroup* valueGroup = cast(KWDRValueGroup*, kwdrGroups->GetOperandAt(i)->GetDerivationRule());

		for (int nValue = 0; nValue < valueGroup->GetValueNumber(); nValue++)
		{
			ALString sLabel = valueGroup->GetValueAt(nValue).GetValue();
			if (sLabel != Symbol::GetStarValue()) {
				StringObject* s = new StringObject();
				s->SetString(sLabel);
				atomicModalities.Add(s);
			}
		}
	}

	// ajout du libellé "unseen values", en gérant la presence eventuelle d'une modalité ayant deja cette valeur.
	// ne pas toucher a l'ordonnancement d'origine des modalités
	ObjectArray* sortedModalities = atomicModalities.Clone();
	sortedModalities->SetCompareFunction(KMCompareLabels);
	sortedModalities->Sort();
	StringObject* s = KMParameters::GetUniqueLabel(*sortedModalities, (supervisedMode ? "Unseen values" : "Other or unseen values"));
	assert(s != NULL);
	atomicModalities.Add(s);

	AddAtomicModalities(attributeName, atomicModalities.Clone());

	delete sortedModalities;

}


void KMAttributesPartitioningManager::AddIntervalBounds(KWDRIntervalBounds* kwdrIntervalBounds, const ALString attributeName) {

	ObjectArray partitions;

	if (kwdrIntervalBounds->GetIntervalBoundNumber() == 0) {
		// cas particulier d'un intervalBound vide
		StringObject* s = new StringObject();
		s->SetString(ALString("]-inf;+inf]"));
		partitions.Add(s);

	}
	else {

		// recuperer les intervalles, aux fins d'affichage
		for (int i = 0; i < kwdrIntervalBounds->GetIntervalBoundNumber(); i++) {

			if (i == 0) {
				StringObject* s = new StringObject();
				s->SetString(ALString("]-inf;") +
					DoubleToString(kwdrIntervalBounds->GetIntervalBoundAt(i)) + "]");
				partitions.Add(s);

				if (i == kwdrIntervalBounds->GetIntervalBoundNumber() - 1) {
					s = new StringObject();
					s->SetString(ALString("]") + DoubleToString(kwdrIntervalBounds->GetIntervalBoundAt(i)) + ALString(";+inf]"));

					partitions.Add(s);
				}

			}
			else {
				StringObject* s = new StringObject();
				s->SetString(ALString("]") + DoubleToString(kwdrIntervalBounds->GetIntervalBoundAt(i - 1)) + ALString(";") +
					DoubleToString(kwdrIntervalBounds->GetIntervalBoundAt(i)) + "]");

				partitions.Add(s);

				if (i == kwdrIntervalBounds->GetIntervalBoundNumber() - 1) {
					s = new StringObject();
					s->SetString(ALString("]") + DoubleToString(kwdrIntervalBounds->GetIntervalBoundAt(i)) + ALString(";+inf]"));

					partitions.Add(s);
				}
			}
		}
	}

	AddAttributePartitions(attributeName, partitions.Clone());

}


KMAttributesPartitioningManager* KMAttributesPartitioningManager::Clone()
{

	KMAttributesPartitioningManager* aClone;

	aClone = new KMAttributesPartitioningManager();
	aClone->CopyFrom(this);

	return aClone;
}

void KMAttributesPartitioningManager::CopyFrom(const KMAttributesPartitioningManager* aSource)
{
	require(aSource != NULL);

	// copier les ObjectArray de modalités groupées ou d'intervalles (auparavant, supprimer les donnees eventuellement existantes)
	POSITION position = odAttributesPartitions.GetStartPosition();
	ALString key;
	Object* oCurrent;

	while (position != NULL) {
		odAttributesPartitions.GetNextAssoc(position, key, oCurrent);
		ObjectArray* oaModalities = cast(ObjectArray*, oCurrent);
		oaModalities->DeleteAll();
		delete oaModalities;
	}

	position = aSource->odAttributesPartitions.GetStartPosition();
	while (position != NULL) {
		aSource->odAttributesPartitions.GetNextAssoc(position, key, oCurrent);
		ObjectArray* oaSourceModalities = cast(ObjectArray*, oCurrent);
		ObjectArray* oaTargetModalities = new ObjectArray;
		for (int i = 0; i < oaSourceModalities->GetSize(); i++) {
			StringObject* so = cast(StringObject*, oaSourceModalities->GetAt(i));
			oaTargetModalities->Add(so->Clone());
		}
		odAttributesPartitions.SetAt(key, oaTargetModalities);
	}

	// idem pour les modalités non groupées :

	position = odAtomicModalities.GetStartPosition();
	while (position != NULL) {
		odAtomicModalities.GetNextAssoc(position, key, oCurrent);
		ObjectArray* oaModalities = cast(ObjectArray*, oCurrent);
		oaModalities->DeleteAll();
		delete oaModalities;
	}

	position = aSource->odAtomicModalities.GetStartPosition();
	while (position != NULL) {
		aSource->odAtomicModalities.GetNextAssoc(position, key, oCurrent);
		ObjectArray* oaSourceModalities = cast(ObjectArray*, oCurrent);
		ObjectArray* oaTargetModalities = new ObjectArray;
		for (int i = 0; i < oaSourceModalities->GetSize(); i++) {
			StringObject* so = cast(StringObject*, oaSourceModalities->GetAt(i));
			oaTargetModalities->Add(so->Clone());
		}
		odAtomicModalities.SetAt(key, oaTargetModalities);
	}
}






