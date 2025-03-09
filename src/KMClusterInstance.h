// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once


#include "Object.h"
#include "KWObject.h"
#include "KMParameters.h"

class KMClusterInstanceAttribute;

//////////////////////
/// classe representant un individu appartenant a un cluster. Il peut, ou non, correspondre a une instance de BDD reelle. Il peut etre construit a partir d'un objet de database (KWObject), mais ne necessite pas la persistance en memoire du KWObject correspondant, une fois construit.
//

class KMClusterInstance : public Object
{
public:

	KMClusterInstance();

	KMClusterInstance(const KWObject*, const KMParameters*);

	~KMClusterInstance(void);

	const ObjectArray& GetLoadedAttributes() const;

	const KMClusterInstanceAttribute* FindAttribute(const ALString nativeName, const ALString recodedName) const;

	const KMClusterInstanceAttribute* FindAttribute(const KWLoadIndex& loadIndex) const;

	Continuous GetContinuousValueAt(const KWLoadIndex& loadIndex) const;

	KMClusterInstance* Clone();

	void CopyFrom(const KMClusterInstance* aSource);

protected:

	void AddLoadedAttributes(const KWObject*);

	const KMParameters* parameters;

	/** ensemble de pointeurs sur KMClusterInstanceAttribute   */
	ObjectArray oaLoadedAttributes;
};


/// classe KMClusterInstanceAttribute : sert a stocker les valeurs "utiles" pour kmean, d'une instance de cluster
class KMClusterInstanceAttribute : public Object
{
public:

	KMClusterInstanceAttribute(KWLoadIndex w_loadIndex, ALString w_nativeName, ALString w_recodedName, Continuous w_continuousValue,
		Symbol w_symbolValue, int w_type) :
		liLoadIndex(w_loadIndex), nativeName(w_nativeName), recodedName(w_recodedName),
		continuousValue(w_continuousValue), symbolicValue(w_symbolValue), type(w_type) {}

	const KWLoadIndex liLoadIndex;
	const ALString nativeName;
	const ALString recodedName;
	const Continuous continuousValue;
	const Symbol symbolicValue;
	const int type;
};

// fonctions de tri/comparaison
int
KMClusterInstanceAttributeSortNativeNameAsc(const void* elem1, const void* elem2);

