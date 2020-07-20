// stdafx.cpp : source file that includes just the standard includes
// SyntheticData.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"
using namespace std;
using namespace Eigen;

extern int NVAR;
extern int NVAR_HID;
extern int degree;
int ** samples;
int ** samples_hid;
int node_counter;
int node_counter_hid;
MatrixXd observable_hidden;
MatrixXd hidden_hidden;
// TODO: reference any additional headers you need in STDAFX.H
// and not in this file
void generate_transition_matrices(int observable_dimension, int hidden_dimension, int column_sparsity, string folder_name)
{
	observable_hidden = MatrixXd::Random(observable_dimension, hidden_dimension);
	hidden_hidden = MatrixXd::Random(hidden_dimension, hidden_dimension);

	observable_hidden.array() += 1; // to make the entries positive, i.e., U(0,2)
	for (int i = 1; i<observable_dimension; i++) // i starts from 1 to make sure there is atleast 1 non-zero element; else model doesn't make sense and also the probability normalization below would lead to divide by zero (or nan) error
		for (int j = 0; j<hidden_dimension; j++)
			if (observable_hidden(i, j) <= 2 * column_sparsity) // overall sparsity = per column sparsity (in expectation)
				observable_hidden(i, j) = 0;
	for (int j = 0; j<hidden_dimension; j++)
		observable_hidden.col(j) /= observable_hidden.col(j).sum();
	
	hidden_hidden.array() += 1; // to make the entries positive, i.e., U(0,2)
	for (int j = 0; j<hidden_dimension; j++)
		hidden_hidden.col(j) /= hidden_hidden.col(j).sum();

	// write transition matrices to file here
	string file_name;
	fstream fout;
	file_name = folder_name;
	fout.open(file_name.append("/observable_hidden.txt").c_str(), ios::out);
	for (int od = 0; od<observable_dimension; od++)
	{
		for (int hd = 0; hd<hidden_dimension; hd++)
		{
			fout << observable_hidden(od, hd) << '\t';
		}
		fout << endl;
	}
	fout.close();

	file_name = folder_name;
	fout.open(file_name.append("/hidden_hidden.txt").c_str(), ios::out);
	for (int od = 0; od<hidden_dimension; od++)
	{
		for (int hd = 0; hd<hidden_dimension; hd++)
		{
			fout << hidden_hidden(od, hd) << '\t';
		}
		fout << endl;
	}
	fout.close();

//	std::cout << "observable_hidden: " << observable_hidden << endl;
//	std::cout << "hidden_hidden: " << hidden_hidden << endl;

	// computing columnwise cdf for sampling
	for (int j = 0; j<hidden_dimension; j++)
		for (int i = 1; i<observable_dimension; i++)
			observable_hidden(i, j) = observable_hidden(i - 1, j) + observable_hidden(i, j);
	for (int j = 0; j<hidden_dimension; j++)
		for (int i = 1; i<hidden_dimension; i++)
			hidden_hidden(i, j) = hidden_hidden(i - 1, j) + hidden_hidden(i, j);

//	std::cout << "observable_hidden: " << observable_hidden << endl;
//	std::cout << "hidden_hidden: " << hidden_hidden << endl;
	

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void sample_tree(int observable_dimension, int hidden_dimension, int column_sparsity, string folder_name, int number_of_samples, int number_of_levels)
{
	std::cout << "Sampling tree" << endl;
	MatrixXd observable_hidden(observable_dimension, hidden_dimension);
	MatrixXd hidden_hidden(hidden_dimension, hidden_dimension);
	generate_transition_matrices(observable_dimension, hidden_dimension, column_sparsity, folder_name);


	

	VectorXd root_node_probability = VectorXd::Ones(hidden_dimension); // MatrixXd::Random(hidden_dimension, 1);//	root_node_probability.array() += 1;
	root_node_probability/= root_node_probability.sum();

	// write root node probability to file here
	string file_name, file_name1, file_name2;
	fstream fout, fout1, fout2;
	file_name = folder_name;
	fout.open(file_name.append("/root_node_probability.txt").c_str(), ios::out);
	for (int rnp = 0; rnp<hidden_dimension; rnp++) // assume we always have atleast 2 levels - else include a check condition
	{
		fout << root_node_probability(rnp) << '\t';
	}
	fout.close();
	for (int i = 1; i < root_node_probability.rows(); i++)
		root_node_probability(i) = root_node_probability(i - 1) + root_node_probability(i);


	samples = zeros(number_of_samples, NVAR);
	samples_hid = zeros(number_of_samples, NVAR_HID);
//sampling
	file_name1 = folder_name;
	fout1.open(file_name1.append("/samples.txt").c_str(), ios::out);

	file_name2 = folder_name;
	fout2.open(file_name2.append("/samples_hid.txt").c_str(), ios::out);

	for (int ns = 0; ns < number_of_samples; ++ns)
	{
		node_counter = 0; node_counter_hid = 0;
		if (ns % 1000== 0)
			std::cout << "sample number = " << ns << endl;
		
		double random_number = (double)rand() / (double)RAND_MAX;
		int root_node_state;

		int low = 0, high = hidden_dimension - 1; // binary search
		while (low != high)
		{
			int mid = (low + high) / 2;
			if (root_node_probability(mid, 0) < random_number)
				low = mid + 1;
			else
				high = mid;
		}
		root_node_state = low; // or high
		samples_hid[ns][node_counter_hid] = root_node_state;
		node_counter_hid++;
		sample_children(root_node_state, 0, observable_dimension, hidden_dimension, number_of_levels, ns); // start sampling
		// write samples and samples_hid to files
		
		if (ns % 1000==0)
			std::cout << "writing sample " << ns << endl;

		for (int nvar = 0; nvar < NVAR; ++nvar)
		{
			fout1 << samples[ns][nvar] << '\t';
			// sampleID node_val1 node_val2 ... node_val{NVAR}
		}
		fout1 << endl;
		

		// groundtruth
		
//		std::cout << "writing sample hid " << ns << endl;
		for (int nvar = 0; nvar < NVAR_HID; ++nvar)
		{
			fout2 << samples_hid[ns][nvar] << '\t';
			// sampleID node_val1 node_val2 ... node_val{NVAR}
		}
		fout2 << endl;
		
	}
	fout1.close();
	fout2.close();
}



void sample_children(int parent_state, int parent_level ,int observable_dimension, int hidden_dimension, int number_of_levels,int ns)
{
	for (int i = 0; i<degree; i++)
	{
		double random_number = (double)rand() / (double)RAND_MAX;
		int current_node_state;
		if (parent_level < number_of_levels - 1)
		{
			// MatrixXd current_node_probability = hidden_hidden.col(parent_state);
			int low = 0, high = hidden_dimension - 1; // binary search
			while (low != high)
			{
				int mid = (low + high) / 2;
				if (hidden_hidden(mid, parent_state) < random_number)
					low = mid + 1;
				else
					high = mid;
			}
			current_node_state = low; // or high
			samples_hid[ns][node_counter_hid] = current_node_state; // update the samples matrix, this is discrete variable, which denotes the categorical variable in an efficient way
			node_counter_hid++;
			sample_children(current_node_state, parent_level + 1, observable_dimension, hidden_dimension, number_of_levels, ns);
		}
		/*
		else if (parent_level == number_of_levels - 1)
		{
			// MatrixXd current_node_probability = observable_hidden.col(parent_state);
			int low = 0, high = observable_dimension - 1; // binary search
			while (low != high)
			{
				int mid = (low + high) / 2;
				if (observable_hidden(mid, parent_state) < random_number)
					low = mid + 1;
				else
					high = mid;
			}
			current_node_state = low; // or high
			samples_hid[ns][node_counter_hid] = current_node_state; // update the samples matrix, this is discrete variable, which denotes the categorical variable in an efficient way
			node_counter_hid++;
			sample_children(current_node_state, parent_level + 1, observable_dimension, hidden_dimension, number_of_levels, ns);
		}
		*/
		else
		{
			double random_number = (double)rand() / (double)RAND_MAX;
			int low = 0, high = observable_dimension - 1; // binary search
			while (low != high)
			{
				int mid = (low + high) / 2;
				if (observable_hidden(mid, parent_state) < random_number)
					low = mid + 1;
				else
					high = mid;
			}
			current_node_state = low; // or high
			samples[ns][node_counter] = current_node_state; // update the samples matrix, this is discrete variable, which denotes the categorical variable in an efficient way
			node_counter++;
//			return;
		}
	}
}



/*
// ground_truth wirte to disk
fstream fout;
string file_name;
file_name = folder_name;
fout.open(file_name.append("/ground_truth.txt").c_str(), ios::out);
for (int o = 0; o < number_of_observable_nodes; o++)
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
*/