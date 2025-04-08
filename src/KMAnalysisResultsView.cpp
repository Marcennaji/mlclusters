// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMAnalysisResultsView.h"


KMAnalysisResultsView::KMAnalysisResultsView()
{
    SetIdentifier("KMAnalysisResults");
    SetLabel("Analysis results");

}


KMAnalysisResultsView::~KMAnalysisResultsView()
{
}


void KMAnalysisResultsView::EventUpdate(Object* object)
{
    KMAnalysisResults* editedObject;

    require(object != NULL);

    KWAnalysisResultsView::EventUpdate(object);
    editedObject = cast(KMAnalysisResults*, object);

}


void KMAnalysisResultsView::EventRefresh(Object* object)
{
    KMAnalysisResults* editedObject;

    require(object != NULL);

    KWAnalysisResultsView::EventRefresh(object);
    editedObject = cast(KMAnalysisResults*, object);

}


const ALString KMAnalysisResultsView::GetClassLabel() const
{
    return "Analysis results";
}

