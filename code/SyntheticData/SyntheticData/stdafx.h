// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
#define _CRT_SECURE_NO_WARNINGS
//	#include "targetver.h"

#include "../../latentTree/dependency/Eigen/Dense"
#include "../../latentTree/dependency/Eigen/Core"



#include <time.h>
#include <stdlib.h>
#include <cmath>
// #include <stdio.h>


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


#include <vector>
using namespace std;
using namespace Eigen;
// TODO: reference additional headers your program requires here
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int furong_atoi(string word);
double furong_atof(string word);
int** zeros(unsigned int r, unsigned int c);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void generate_transition_matrices(int observable_dimension, int hidden_dimension, int column_sparsity, string folder_name);
void sample_tree(int observable_dimension, int hidden_dimension, int column_sparsity, string folder_name, int number_of_samples, int number_of_levels);
void sample_children(int parent_state, int parent_level, int observable_dimension, int hidden_dimension, int number_of_levels, int ns);