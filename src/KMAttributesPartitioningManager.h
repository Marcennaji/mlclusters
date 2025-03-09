// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWObject.h"
#include "KWClassStats.h"


///  classe gerant les intervalles et modalites (groupees ou non) des attributs

class KMAttributesPartitioningManager : public Object
{
public:

	KMAttributesPartitioningManager();
	~KMAttributesPartitioningManager(void);

	KMAttributesPartitioningManager* Clone();

	void CopyFrom(const KMAttributesPartitioningManager* aSource);

	/** ajout des groupes de valeurs d'un attribut, a partir d'une regle de derivation */
	void AddValueGroups(KWDRValueGroups* kwdrGroups, const ALString attributeName, const int maxValuesToAdd, const bool supervisedMode);

	/** ajout des intervalles de valeurs d'un attribut, a partir d'une regle de derivation */
	void AddIntervalBounds(KWDRIntervalBounds* kwdrIntervalBounds, const ALString attributeName);

	/** cle = nom d'attribut, Valeur =  ObjectArray de modalités ou d'intervalles*/
	const ObjectDictionary& GetPartitions() const;

	/** cle = nom d'attribut. Valeur = ObjectArray de modalités non groupées */
	const ObjectDictionary& GetAtomicModalities() const;


protected:

	/** memorise un tableau des modalites groupées/intervalles, pour un attribut donné */
	void AddAttributePartitions(const ALString attributeName, ObjectArray* partitions);

	/** memorise un tableau des modalites non groupées, pour un attribut donné */
	void AddAtomicModalities(const ALString attributeName, ObjectArray* modalities);

	// ==============================================  attributs de la classe ===============================================

	/**
	Clé = nom de l'attribut, valeur = ObjectArray * contenant des StringObject * --> liste de toutes les modalités groupees ou intervalles d'un attribut*/
	ObjectDictionary odAttributesPartitions;

	/**
	Clé = nom de l'attribut, valeur = ObjectArray * contenant des StringObject * --> liste de toutes les modalités non groupées ('atomiques') d'un attribut */
	ObjectDictionary odAtomicModalities;

};




