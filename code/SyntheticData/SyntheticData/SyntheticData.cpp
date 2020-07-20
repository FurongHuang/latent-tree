// SyntheticData.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"




using namespace std;
using namespace Eigen;

// made global for stack (due to recursive function call) and time efficiency; some can be #defined
//	string folder_name = "tree_1";
//	int observable_dimension = 100;
//	int VOCA_SIZE = 10;
//	double column_sparsity = 0.6; // average sparsity level in [0,1];
//	int number_of_levels = 2;
//	const int NX = 150000;
int degree = 2;

int NVAR;
int NVAR_HID;




int main(int argc, const char * argv[])
{	// C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic 3 2 2 0 100000
	string folder_name = argv[1];							//C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic
	int VOCA_SIZE = furong_atoi(argv[2]);					//3
	int KHID = furong_atoi(argv[3]);						//2
	int number_of_levels = furong_atoi(argv[4]);			//2
	double column_sparsity = furong_atof(argv[5]);			//0
	int NX = furong_atoi(argv[6]);							//100000

	NVAR = (int) pow(degree, number_of_levels);
	NVAR_HID = (int)pow(degree, number_of_levels + 1) - 1 - NVAR;

	fstream fout;
	string mkdir = "mkdir ";
	system(mkdir.append(folder_name).c_str());
	string file_name;


	sample_tree(VOCA_SIZE, KHID, column_sparsity, folder_name, NX, number_of_levels);

	file_name = folder_name;
	fout.open(file_name.append("/NVAR.txt").c_str(), ios::out);
	fout << NVAR << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/number_of_levels.txt").c_str(), ios::out);
	fout << number_of_levels << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/NX.txt").c_str(), ios::out);
	fout << NX << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/degree.txt").c_str(), ios::out);
	fout << degree << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/KHID.txt").c_str(), ios::out);
	fout << KHID << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/VOCA_SIZE.txt").c_str(), ios::out);
	fout << VOCA_SIZE << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/column_sparsity.txt").c_str(), ios::out);
	fout << column_sparsity << endl;
	fout.close();

	srand(time(NULL));

	return 0;
}