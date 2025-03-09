// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMClusterInstance.h"


KMClusterInstance::KMClusterInstance(){

}

KMClusterInstance::~KMClusterInstance(void){

	oaLoadedAttributes.DeleteAll();
}


KMClusterInstance::KMClusterInstance(const KWObject * _instance, const KMParameters * _parameters) : parameters(_parameters){

	AddLoadedAttributes(_instance);
}

void KMClusterInstance::AddLoadedAttributes(const KWObject * kwo){

	for (int j = 0; j < kwo->GetClass()->GetLoadedAttributeNumber(); j++){

		KWAttribute * attribute = kwo->GetClass()->GetLoadedAttributeAt(j);

		ALString nativeAttributeName;
		ALString recodedAttributeName;

		if (parameters->GetNativeAttributeName(attribute->GetName()) == ""){
			// l'attribut recodé correspondant n'a pas été trouvé : c'est donc un attribut natif
			nativeAttributeName = attribute->GetName();
			recodedAttributeName = "";
		}
		else{
			// c'est un attribut recodé
			nativeAttributeName = parameters->GetNativeAttributeName(attribute->GetName());
			recodedAttributeName = attribute->GetName();
		}

		KMClusterInstanceAttribute * value = NULL;

		if (attribute->GetType() == KWType::Continuous)
			value = new KMClusterInstanceAttribute(attribute->GetLoadIndex(), nativeAttributeName, recodedAttributeName, kwo->GetContinuousValueAt(attribute->GetLoadIndex()), "", KWType::Continuous);
		else
		if (attribute->GetType() == KWType::Symbol)
			value = new KMClusterInstanceAttribute(attribute->GetLoadIndex(), nativeAttributeName, recodedAttributeName, -1, kwo->GetSymbolValueAt(attribute->GetLoadIndex()), KWType::Symbol);

		if (value != NULL)
			oaLoadedAttributes.Add(value);

	}
	// tri ascendant sur le nom de l'attribut natif
	oaLoadedAttributes.SetCompareFunction(KMClusterInstanceAttributeSortNativeNameAsc);
	oaLoadedAttributes.Sort();
}

const ObjectArray & KMClusterInstance::GetLoadedAttributes() const{

	return oaLoadedAttributes;
}

const KMClusterInstanceAttribute *
KMClusterInstance::FindAttribute(const ALString nativeName, const ALString recodedName) const {

	for (int i=0; i < oaLoadedAttributes.GetSize(); i++){

		KMClusterInstanceAttribute * attribute = cast(KMClusterInstanceAttribute *, oaLoadedAttributes.GetAt(i));

		if (attribute->nativeName == nativeName and attribute->recodedName == recodedName )
			return attribute;
	}

	return NULL;
}

const KMClusterInstanceAttribute *
KMClusterInstance::FindAttribute(const KWLoadIndex & loadIndex) const {

	assert(loadIndex.IsValid());

	for (int i = 0; i < oaLoadedAttributes.GetSize(); i++){

		KMClusterInstanceAttribute * attribute = cast(KMClusterInstanceAttribute *, oaLoadedAttributes.GetAt(i));

		if (attribute->liLoadIndex == loadIndex)
			return attribute;
	}

	return NULL;
}
Continuous KMClusterInstance::GetContinuousValueAt(const KWLoadIndex & loadIndex) const{

	const KMClusterInstanceAttribute * attribute = FindAttribute(loadIndex);
	assert(attribute != NULL);
	assert(attribute->type == KWType::Continuous);
	return attribute->continuousValue;
}

KMClusterInstance* KMClusterInstance::Clone()
{

    KMClusterInstance* aClone = new KMClusterInstance();
    aClone->CopyFrom(this);
    return aClone;
}


void KMClusterInstance::CopyFrom(const KMClusterInstance * aSource)
{
    require(aSource != NULL);

	oaLoadedAttributes.DeleteAll();

	for (int i=0; i < aSource->oaLoadedAttributes.GetSize(); i++){

		KMClusterInstanceAttribute * origin = cast(KMClusterInstanceAttribute * , aSource->oaLoadedAttributes.GetAt(i));

		KMClusterInstanceAttribute * dest = new KMClusterInstanceAttribute(origin->liLoadIndex, origin->nativeName,
																			origin->recodedName, origin->continuousValue,
																			origin->symbolicValue, origin->type);

		oaLoadedAttributes.Add(dest);
	}

	parameters = aSource->parameters;
}

int
KMClusterInstanceAttributeSortNativeNameAsc(const void* elem1, const void* elem2 )
{
	KMClusterInstanceAttribute * attr1 = (KMClusterInstanceAttribute*) *(Object**)elem1;
	KMClusterInstanceAttribute * attr2 = (KMClusterInstanceAttribute*) *(Object**)elem2;

    // Comparaison de 2 attributs sur le nom de variable natif
	ALString s1(attr1->nativeName);
	s1.MakeLower();

	ALString s2(attr2->nativeName);
	s2.MakeLower();

	return (s2 > s1 ? -1 : 1);
}

