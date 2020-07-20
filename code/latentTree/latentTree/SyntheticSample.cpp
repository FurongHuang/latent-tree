#pragma once
#include "stdafx.h"
extern const int KHID;
extern const int NX;
extern const int NVAR;
extern const int VOCA_SIZE;
extern const double edgeD_MAX;
using namespace Eigen;
using namespace std;


// int VOCA_SIZE = 100;
// int KHID = 10;
double column_sparsity = 0; // average sparsity level in [0,1];

// const int NX = 150000;
int degree = 2;
int ns;
int node_counter;
MatrixXd observable_hidden, hidden_hidden;
int number_of_observable_nodes = (int)pow(degree, number_of_levels - 1);
int ** samples;// [NVAR][NX]; // first argument must be at least the number of observable nodes; note - this might need -mcmodel=large at compile time

static int** zeros(unsigned int r, unsigned int c)
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

void ground_truth(string folder_name)
{
	fstream fout;
	string file_name;
	file_name = folder_name;
	fout.open(file_name.append("/ground_truth.txt").c_str(), ios::out);
	for (int o = 0; o<number_of_observable_nodes; o++)
	{
		int rem = (number_of_observable_nodes - 1) / (degree - 1);
		rem += o;
		cout << "remainder  = " << rem << endl;
		while (rem>0)
		{
			rem = (rem - 1) / degree;
			fout << o << '\t' << rem << '\t' << 1 << endl;
		}
	}
	fout.close();
}

void sample_children(int parent_state, int parent_level,int ** samples)
{
	
	for (int i = 0; i<degree; i++)
	{
		double random_number = (double)rand() / (double)RAND_MAX;
		int current_node_state;
		if (parent_level < number_of_levels - 2)
		{
			// MatrixXd current_node_probability = hidden_hidden.col(parent_state);
			int low = 0, high = KHID - 1; // binary search
			while (low != high)
			{
				int mid = (low + high) / 2;
				if (hidden_hidden(mid, parent_state) < random_number)
					low = mid + 1;
				else
					high = mid;
			}
			current_node_state = low; // or high

			/*for(int s=0; s<KHID; s++) // linear search
			if(random_number < hidden_hidden(s, parent_state))
			{
			current_node_state = s;
			break;
			}*/

			sample_children(current_node_state, parent_level + 1);
		}
		else if (parent_level == number_of_levels - 2)
		{
			// MatrixXd current_node_probability = observable_hidden.col(parent_state);
			int low = 0, high = VOCA_SIZE - 1; // binary search
			while (low != high)
			{
				int mid = (low + high) / 2;
				if (observable_hidden(mid, parent_state) < random_number)
					low = mid + 1;
				else
					high = mid;
			}
			current_node_state = low; // or high

			/*for(int s=0; s<VOCA_SIZE; s++) // linear search
			if(random_number < observable_hidden(s, parent_state))
			{
			current_node_state = s;
			break;
			}*/
			sample_children(current_node_state, parent_level + 1);
		}
		else
		{
			current_node_state = parent_state;
			samples[node_counter][ns] = current_node_state; // update the samples matrix
			node_counter++;
			return;
		}
	}
}

void generate_transition_matrices(string folder_name)
{
	observable_hidden = MatrixXd::Random(VOCA_SIZE, KHID);
	observable_hidden.array() += 1; // to make the entries positive, i.e., U(0,2)
	for (int i = 1; i<VOCA_SIZE; i++) // i starts from 1 to make sure there is atleast 1 non-zero element; else model doesn't make sense and also the probability normalization below would lead to divide by zero (or nan) error
		for (int j = 0; j<KHID; j++)
			if (observable_hidden(i, j) <= 2 * column_sparsity) // overall sparsity = per column sparsity (in expectation)
				observable_hidden(i, j) = 0;
	for (int j = 0; j<KHID; j++)
		observable_hidden.col(j) /= observable_hidden.col(j).sum();
	hidden_hidden = MatrixXd::Random(KHID, KHID);
	hidden_hidden.array() += 1; // to make the entries positive, i.e., U(0,2)
	for (int j = 0; j<KHID; j++)
		hidden_hidden.col(j) /= hidden_hidden.col(j).sum();
	// write transition matrices to file here
	string file_name;
	fstream fout;
	file_name = folder_name;
	fout.open(file_name.append("/observable_hidden.txt").c_str(), ios::out);
	for (int od = 0; od<VOCA_SIZE; od++)
	{
		for (int hd = 0; hd<KHID; hd++)
		{
			fout << observable_hidden(od, hd) << '\t';
		}
		fout << endl;
	}
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/hidden_hidden.txt").c_str(), ios::out);
	for (int od = 0; od<KHID; od++)
	{
		for (int hd = 0; hd<KHID; hd++)
		{
			fout << hidden_hidden(od, hd) << '\t';
		}
		fout << endl;
	}
	fout.close();

	// computing columnwise cdf for sampling
	for (int j = 0; j<KHID; j++)
		for (int i = 1; i<VOCA_SIZE; i++)
			observable_hidden(i, j) = observable_hidden(i - 1, j) + observable_hidden(i, j);
	for (int j = 0; j<KHID; j++)
		for (int i = 1; i<KHID; i++)
			hidden_hidden(i, j) = hidden_hidden(i - 1, j) + hidden_hidden(i, j);
}

void sample_tree(string folder_name)
{
	cout << "Sampling tree" << endl;
	generate_transition_matrices();
	MatrixXd root_node_probability = MatrixXd::Ones(KHID, 1); // MatrixXd::Random(KHID, 1);
	root_node_probability.array() += 1;
	root_node_probability.col(0) /= root_node_probability.col(0).sum();
	// write root node probability to file here
	string file_name;
	fstream fout;
	file_name = folder_name;
	fout.open(file_name.append("/root_node_probability.txt").c_str(), ios::out);
	for (int rnp = 0; rnp<KHID; rnp++) // assume we always have atleast 2 levels - else include a check condition
	{
		fout << root_node_probability(rnp, 0) << '\t';
	}
	fout.close();


	for (ns = 0; ns<NX; ns++)
	{
		node_counter = 0;
		cout << "sample number = " << ns << endl;
		for (int i = 1; i<root_node_probability.rows(); i++)
			root_node_probability(i, 0) = root_node_probability(i - 1, 0) + root_node_probability(i, 0);
		double random_number = (double)rand() / (double)RAND_MAX;
		int root_node_state;

		int low = 0, high = KHID - 1; // binary search
		while (low != high)
		{
			int mid = (low + high) / 2;
			if (root_node_probability(mid, 0) < random_number)
				low = mid + 1;
			else
				high = mid;
		}
		root_node_state = low; // or high

		/*for(int s=0; s<KHID; s++) // linear search
		if(random_number < root_node_probability(s, 0))
		{
		root_node_state = s;
		break;
		}*/

		sample_children(root_node_state, 0); // start sampling
	}
}


int syntheticSampleGen(string folder_name, int number_of_levels, )
{
	samples = zeros(NVAR,NX);
	fstream fout;
	string mkdir = "mkdir ";
	system(mkdir.append(folder_name).c_str());
	string file_name;

	file_name = folder_name;
	fout.open(file_name.append("/number_of_observable_nodes.txt").c_str(), ios::out);
	fout << number_of_observable_nodes << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/number_of_levels.txt").c_str(), ios::out);
	fout << number_of_levels << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/number_of_samples.txt").c_str(), ios::out);
	fout << number_of_samples << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/degree.txt").c_str(), ios::out);
	fout << degree << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/observable_dimension.txt").c_str(), ios::out);
	fout << observable_dimension << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/hidden_dimension.txt").c_str(), ios::out);
	fout << hidden_dimension << endl;
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/column_sparsity.txt").c_str(), ios::out);
	fout << column_sparsity << endl;
	fout.close();

	srand(time(NULL));
	sample_tree();

	file_name = folder_name;
	fout.open(file_name.append("/samples.txt").c_str(), ios::out);
	for (int lnn = 0; lnn<number_of_observable_nodes; lnn++)
	{
		cout << "writing node = " << lnn << endl;
		for (int lns = 0; lns<number_of_samples; lns++)
		{
			fout << lns << '\t' << samples[lnn][lns] << '\t' << 1 << endl;
			// fout << lnn << '\t' << lns << '\t' << samples[lnn][lns] << '\t' << 1 << endl;
		}
		fout << -1 << endl;
	}
	fout.close();

	ground_truth();

	return 0;
}