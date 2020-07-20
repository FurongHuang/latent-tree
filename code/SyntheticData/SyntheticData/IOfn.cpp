
#include "stdafx.h"

int furong_atoi(string word)
{
	int lol = atoi(word.c_str()); /*c_str is needed to convert string to const char*
								  previously (the function requires it)*/
	return lol;
}

double furong_atof(string word)
{
	double lol = atof(word.c_str()); /*c_str is needed to convert string to const char*
									 previously (the function requires it)*/
	return lol;
}

int** zeros(unsigned int r, unsigned int c)
{
	int** rv = (int**)malloc(r * sizeof(int*));
	assert(rv != NULL);

	for (unsigned int i = 0; i < r; ++i)
	{
		rv[i] = (int*)calloc(c, sizeof(int));
		assert(rv[i] != NULL);
	}

	return rv;
}