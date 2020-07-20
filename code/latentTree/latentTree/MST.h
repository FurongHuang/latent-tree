#ifndef __latentTree__MST__
#define __latentTree__MST__
#include "stdafx.h"

vector<Edge*> BoruvkaMST(vector<Edge> EV, vector<Node> mynodelist);
vector<Edge*> KruskalMST(vector<Edge> &EV, vector<Node> &mynodelist);
#endif