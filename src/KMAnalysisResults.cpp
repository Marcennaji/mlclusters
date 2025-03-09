// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMAnalysisResults.h"


KMAnalysisResults::KMAnalysisResults()
{
}


KMAnalysisResults::~KMAnalysisResults()
{

}


void KMAnalysisResults::CopyFrom(const KMAnalysisResults* aSource)
{
    require(aSource != NULL);

    KWAnalysisResults::CopyFrom(aSource);
}


KMAnalysisResults* KMAnalysisResults::Clone() const
{
    KMAnalysisResults* aClone;

    aClone = new KMAnalysisResults;
    aClone->CopyFrom(this);

    return aClone;
}

void KMAnalysisResults::Write(ostream& ost) const
{
    KWAnalysisResults::Write(ost);

}


const ALString KMAnalysisResults::GetClassLabel() const
{
    return "Analysis results";
}


//## Method implementation

const ALString KMAnalysisResults::GetObjectLabel() const
{
    ALString sLabel;

    return sLabel;
}
