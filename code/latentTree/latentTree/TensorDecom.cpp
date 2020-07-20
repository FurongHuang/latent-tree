#pragma once
#include "stdafx.h"
extern int KHID;
extern int VOCA_SIZE;
extern int NX;
extern double alpha0;
using namespace Eigen;
using namespace std;

// set of whitening matrix
void second_whiten(SparseMatrix<double> Gx_a, SparseMatrix<double> Gx_b, SparseMatrix<double> Gx_c, \
	SparseMatrix<double> &W, SparseMatrix<double> &Z_B, SparseMatrix<double> &Z_C)
{
	double nx = (double)Gx_a.rows();
	double inv_nx;
#ifdef NORMALIZE
	inv_nx = 1 / nx;
#else
	inv_nx = 1;
#endif
	SparseVector<double> my_ones = (VectorXd::Ones((int)nx)).sparseView();
	SparseVector<double> mu_a_sparse = my_ones.transpose() * Gx_a;	SparseVector<double> mu_b_sparse = my_ones.transpose() * Gx_b;	SparseVector<double> mu_c_sparse = my_ones.transpose() * Gx_c;
	mu_a_sparse = inv_nx * mu_a_sparse;	mu_b_sparse = inv_nx * mu_b_sparse;	mu_c_sparse = inv_nx * mu_c_sparse;


	//    cout << "--------------starting to calculate Z_B, Z_C implicitly---------------------------"<< endl;
	SparseMatrix<double> Z_B_numerator = Gx_a.transpose() * Gx_c;// NA * NC
	Z_B_numerator = inv_nx * Z_B_numerator;       // NA * NC
	SparseMatrix<double> Z_B_denominator = Gx_b.transpose()*Gx_c;// NB * NC
	Z_B_denominator = inv_nx * Z_B_denominator;//NB * NC

	SparseMatrix<double> Z_C_numerator = Gx_a.transpose() * Gx_b; // NA * NB
	Z_C_numerator = inv_nx *  Z_C_numerator;       // NA * NB
	SparseMatrix<double> Z_C_denominator = Gx_c.transpose()*Gx_b;//NC * NB
	Z_C_denominator = inv_nx * Z_C_denominator;//NC * NB

	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > V_invL_Utrans_Zc = pinv_Nystrom_sparse_component(Z_C_denominator);
	//X_pinv = V_invL_Utrans.first.first * V_invL_Utrans.second * V_invL_Utrans.first.second;
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > V_invL_Utrans_Zb = pinv_Nystrom_sparse_component(Z_B_denominator);
	
	//    cout << "--------------starting to calculate Z_B, Z_C ---------------------------"<< endl;
	// compute M2 Main Term
	double para_main = inv_nx * (alpha0 + 1);
	// NA by k term: 
	SparseMatrix<double> term_na_k = (Z_C_numerator * V_invL_Utrans_Zc.first.first) * V_invL_Utrans_Zc.second;
	Z_C = term_na_k * V_invL_Utrans_Zc.first.second;
	// k by k term: 
	SparseMatrix<double> term_k_k = (V_invL_Utrans_Zc.first.second * Z_C_denominator) * V_invL_Utrans_Zb.first.second.transpose();

	// k by NA term: 
	SparseMatrix<double> term_k_na = (V_invL_Utrans_Zb.second * V_invL_Utrans_Zb.first.first.transpose()) * Z_B_numerator.transpose();
	Z_B = term_k_na.transpose() * V_invL_Utrans_Zb.first.second;

	SparseMatrix<double> M2 = (term_na_k * term_k_k) * term_k_na; M2 = para_main * M2;
	M2.makeCompressed();	M2.prune(TOLERANCE);

	// compute M2 Shift Term
	double para_shift = alpha0;
	//    cout <<"-------------------computing square_mu_a_sparse--------"<<endl;
	double para = (double)alpha0 / ((double)alpha0 + 1);
	SparseMatrix<double> shiftTerm = mu_a_sparse * mu_a_sparse.transpose(); shiftTerm = para_shift * shiftTerm;
	SparseMatrix<double> M2_a = M2 - shiftTerm;	M2_a.makeCompressed(); M2_a.prune(TOLERANCE);
	//    cout << "-----------M2_alpha0:nonZeros()" << M2.nonZeros()<< "-------------"<<endl;
	
	pair< SparseMatrix<double>, SparseVector<double> > Vw_Lw = SVD_symNystrom_sparse(M2_a);
	SparseMatrix<double> Uw = Vw_Lw.first.leftCols(KHID);
	VectorXd Lw = (VectorXd)Vw_Lw.second;
	Lw = pinv_vector(Lw.head(KHID).cwiseSqrt());
	MatrixXd diag_Lw_sqrt_inv = Lw.asDiagonal();
	SparseMatrix<double> diag_Lw_sqrt_inv_s = diag_Lw_sqrt_inv.sparseView();
	W.resize(Gx_a.cols(), KHID);
	W = Uw * diag_Lw_sqrt_inv_s;
	W.makeCompressed(); W.prune(TOLERANCE);
//	cout << "---------------------dimension of W : " << W.rows() << " , " << W.cols() << "----------------" << endl;
//	cout << "-----------End of Whitening----------nonZeros() of W : " << W.nonZeros() << endl;

}


// tensorDecom tested
pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > tensorDecom(Node * x1, Node * x2, Node * x3)
{

	SparseMatrix<double> whitening; whitening.resize(x3->readsamples().cols(), KHID); whitening.makeCompressed();
	// Z_B na by nb which is sym_x1
	// Z_C na by nc which is sym_x2
	SparseMatrix<double> sym_x1; sym_x1.resize(x3->readsamples().cols(), x1->readsamples().cols()); sym_x1.makeCompressed();
	SparseMatrix<double> sym_x2; sym_x2.resize(x3->readsamples().cols(), x2->readsamples().cols()); sym_x2.makeCompressed();
	cout << "-----------------------whitening starts-----------------------------------" << endl;
	// Doing the unshifted whitening

	second_whiten(x3->readsamples(), x1->readsamples(), x2->readsamples(), whitening, sym_x1, sym_x2);

	cout << "----------------------whitening ends ---------------------------------" << endl;
	cout << "whitening dimension: " << whitening.rows() << " , " << whitening.cols() << " nonzeros:  " << whitening.nonZeros() << "-------------------" << endl;

	// whitening step
	VectorXd lambda(KHID);
	MatrixXd eigenvec(KHID, KHID);

	cout << "---------starts of STGD-----------------------" << endl;

	if (whitening.nonZeros()<KHID + 1){
		lambda = VectorXd::Ones(KHID);
		eigenvec = lambda.asDiagonal();
	}
	else
	{
		//whitening data

		SparseMatrix<double> D3_mat = whitening.transpose() * x3->readsamples().transpose();


		SparseMatrix<double> D1_mat = whitening.transpose() * sym_x1 * x1->readsamples().transpose();


		SparseMatrix<double> D2_mat = whitening.transpose() * sym_x2 * x2->readsamples().transpose();

		//    cout << "---------------Data dimension: " << D3_mat.rows() << " , "<< D3_mat.cols() <<" nonzeros: "<<D3_mat.nonZeros()<<"------------------"<< endl;
		double nx = (double)D3_mat.cols();	double inv_nx = 1 / nx;
		SparseVector<double> my_ones = (VectorXd::Ones((int)nx)).sparseView();
		
		SparseVector<double> mu_D3_mat = D3_mat * my_ones;	mu_D3_mat = inv_nx * mu_D3_mat;
		SparseVector<double> mu_D1_mat = D1_mat * my_ones;	mu_D1_mat = inv_nx * mu_D1_mat;
		SparseVector<double> mu_D2_mat = D2_mat * my_ones;	mu_D2_mat = inv_nx * mu_D2_mat;


		tensorDecom_alpha0(D3_mat, mu_D3_mat, D1_mat, mu_D1_mat, D2_mat, mu_D2_mat,lambda, eigenvec);
	
	}
	cout << "-------------------------------STGD finished --------------------------------" << endl;

	Eigen::VectorXd p_h_dense = normProbVector(lambda.array().pow(-2));
	Eigen::SparseVector<double> p_h; p_h.resize(KHID); p_h = p_h_dense.sparseView(); p_h.prune(TOLERANCE);

	Eigen::SparseMatrix<double> p_x1_h; p_x1_h.resize(x1->readsamples().cols(), KHID);
	Eigen::SparseMatrix<double> p_x2_h; p_x2_h.resize(x2->readsamples().cols(), KHID);
	Eigen::SparseMatrix<double> p_x3_h; p_x3_h.resize(x3->readsamples().cols(), KHID);


	if (whitening.nonZeros() < KHID + 1)
	{
		MatrixXd tmp_p_x1_h = MatrixXd::Ones((x1->readsamples()).cols(), KHID);
		SparseMatrix<double> tmp_p_x1_h_sparse = tmp_p_x1_h.sparseView();
		p_x1_h = normProbMatrix(tmp_p_x1_h_sparse);
		MatrixXd tmp_p_x2_h = MatrixXd::Ones((x2->readsamples()).cols(), KHID);
		SparseMatrix<double> tmp_p_x2_h_sparse = tmp_p_x2_h.sparseView();
		p_x2_h = normProbMatrix(tmp_p_x2_h_sparse);
		MatrixXd tmp_p_x3_h = MatrixXd::Ones((x3->readsamples()).cols(), KHID);
		SparseMatrix<double> tmp_p_x3_h_sparse = tmp_p_x3_h.sparseView();
		p_x3_h = normProbMatrix(tmp_p_x3_h_sparse);

	}
	else
	{
		Eigen::MatrixXd Lambda_diag = lambda.asDiagonal();
		Eigen::SparseMatrix<double> Lambda_diag_sparse;
		Lambda_diag_sparse.resize(KHID, KHID);
		Lambda_diag_sparse = Lambda_diag.sparseView();
		Lambda_diag_sparse.makeCompressed();
		Lambda_diag_sparse.prune(TOLERANCE);

		Eigen::SparseMatrix<double> eigenvec_sparse;
		eigenvec_sparse.resize(KHID, KHID);
		eigenvec_sparse = eigenvec.sparseView();
		eigenvec_sparse.makeCompressed();
		eigenvec_sparse.prune(TOLERANCE);


		//
		SparseMatrix<double> pair_pinv_tmp1 = pinv_aNystrom_sparse((SparseMatrix<double>)(whitening.transpose()*sym_x1));// this should be k * VOCA_SIZE
		SparseMatrix<double> pair_pinv_tmp2 = pinv_aNystrom_sparse((SparseMatrix<double>)(whitening.transpose()*sym_x2));
		SparseMatrix<double> pair_pinv_tmp3 = pinv_aNystrom_sparse((SparseMatrix<double>)(whitening.transpose()));
		//
		//	cout << "first"<< endl;
		//	cout << "pair_pinv_tmp1.rows():"<< pair_pinv_tmp1.rows() << ",pair_pinv_tmp1.cols():"<< pair_pinv_tmp1.cols() << endl;
		//	cout << "pair_pinv_tmp2.rows():"<< pair_pinv_tmp2.rows() << ",pair_pinv_tmp2.cols():"<< pair_pinv_tmp2.cols() << endl;
		//	cout << "pair_pinv_tmp3.rows():"<< pair_pinv_tmp3.rows() << ",pair_pinv_tmp3.cols():"<< pair_pinv_tmp3.cols() << endl;
		//	cout << "sym_x1.rows():" << sym_x1.rows() << "sym_x1.cols()" << sym_x1.cols() << endl;
		//	cout << "sym_x2.rows():" << sym_x2.rows() << "sym_x2.cols()" << sym_x2.cols() << endl;
		//	cout << "whitening.rows(): " << whitening.rows() << ", whitening.cols(): " << whitening.cols() << endl;
		//	cout << "second"<< endl;
		//	cout << "pair_pinv_tmp1.rows():"<< pair_pinv_tmp1.second.rows() << ",pair_pinv_tmp1.cols():"<< pair_pinv_tmp1.second.cols() << endl;
		//	cout << "pair_pinv_tmp2.rows():"<< pair_pinv_tmp2.second.rows() << ",pair_pinv_tmp2.cols():"<< pair_pinv_tmp2.second.cols() << endl;
		//	cout << "pair_pinv_tmp3.rows():"<< pair_pinv_tmp3.second.rows() << ",pair_pinv_tmp3.cols():"<< pair_pinv_tmp3.second.cols() << endl;

		//SparseMatrix<double> tmp1_tmp3 = pair_pinv_tmp1 * sym_x1.transpose();
		//Eigen::SparseMatrix<double> tmp1 = tmp1_tmp3 * whitening;


		//SparseMatrix<double> tmp2_tmp3 = pair_pinv_tmp2 * sym_x2.transpose();
		//Eigen::SparseMatrix<double> tmp2 = tmp2_tmp3 * whitening;


		//Eigen::SparseMatrix<double> tmp3 = pair_pinv_tmp3 * whitening;
		//	cout << "tmp1.rows():"<< tmp1.rows() << ",tmp1.cols():"<< tmp1.cols() << endl;
		//	cout << "tmp2.rows():"<< tmp2.rows() << ",tmp2.cols():"<< tmp2.cols() << endl;
		//	cout << "tmp3.rows():"<< tmp3.rows() << ",tmp3.cols():"<< tmp3.cols() << endl;
		//	cout << "eigenvec_sparse.rows():" << eigenvec_sparse.rows() << "eigenvec_sparse.cols():" << eigenvec_sparse.cols() << endl;
		//	cout << "Lam_diag_sparse.rows():" << Lambda_diag_sparse.rows() << "Lam_diag_sparse.cols():" << Lambda_diag_sparse.cols() << endl;
		// U1 = pinv_matrix(whitening.transpose()*sym_x1)*eigenvec*Lambda_diag;
			p_x1_h = normProbMatrix((SparseMatrix<double>)(pair_pinv_tmp1*eigenvec_sparse * Lambda_diag_sparse));
			p_x2_h = normProbMatrix((SparseMatrix<double>)(pair_pinv_tmp2*eigenvec_sparse * Lambda_diag_sparse));
			p_x3_h = normProbMatrix((SparseMatrix<double>)(pair_pinv_tmp3*eigenvec_sparse * Lambda_diag_sparse));
	}
	pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > output;
#ifdef condi_prob
	output.first.push_back(p_x1_h);
	output.first.push_back(p_x2_h);
	output.first.push_back(p_x3_h);
	output.second = p_h;
#endif

#ifdef joint_prob
	output.first.push_back(condi2condi(P_x1_h, p_h));
	output.first.push_back(condi2condi(P_x2_h, p_h));
	output.first.push_back(condi2condi(P_x3_h, p_h));
	output.second = p_h;
#endif
//	cout << "end tensorDecom function" << endl;;
	//    cout << "-----end of parameter estimation for current triplet ----------"<< endl;
	return output;


}


///////////////////
void tensorDecom_alpha0(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, SparseMatrix<double> D_b_mat, VectorXd D_b_mu, SparseMatrix<double> D_c_mat, VectorXd D_c_mu, VectorXd &lambda, MatrixXd & phi_new)
{
//	cout << "start tensorDecom_alpha0 function" << endl;
	double error;
	MatrixXd A_random(MatrixXd::Random(KHID, KHID));
	MatrixXd phi_old;

	A_random.setRandom();
	HouseholderQR<MatrixXd> qr(A_random);
	phi_new = qr.householderQ();
	A_random.resize(0, 0);

	long iteration = 0;
	while (true)
	{
		long iii = iteration % NX;
		VectorXd D_a_g = D_a_mat.col((int)iii);//
		VectorXd D_b_g = D_b_mat.col((int)iii);
		VectorXd D_c_g = D_c_mat.col((int)iii);

		phi_old = phi_new;
		//Data_a_g = Data_a_G(:,i);Data_b_g = Data_b_G(:,i);Data_c_g = Data_c_G(:,i);
		phi_new = Diff_Loss(D_a_g, D_b_g, D_c_g, D_a_mu, D_b_mu, D_c_mu, phi_old);
		///////////////////////////////////////////////
		if (iteration < MINITER)
		{
		}
		else
		{
			error = (phi_new - phi_old).norm();
			if (error < TOLERANCE || iteration > MAXITER)
			{

				break;

			}
		}

		iteration++;
	}
	Eigen::MatrixXd tmp_lambda = (((phi_new.array().pow(2)).colwise().sum()).pow(3.0 / 2.0)).transpose();
	lambda = tmp_lambda;
	lambda.normalize();
	phi_new = normc(phi_new);
//	cout << "end tensorDecom_alpha0 function" << endl;
}

MatrixXd Diff_Loss(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, Eigen::MatrixXd phi)
{
	MatrixXd New_Phi;

	MatrixXd myvector = MatrixXd::Zero(KHID, KHID);

	for (int index_k = 0; index_k < KHID; index_k++)
	{
		MatrixXd one_eigenvec(KHID, 1);
		one_eigenvec = phi.col(index_k);
		// cout << "one_eigenvec:\n " << one_eigenvec << endl;
		// cout << "phi:\n" << phi << endl;
		//VectorXd tmp = (one_eigenvec.transpose()*phi).array().pow(2);
		MatrixXd tmp = (one_eigenvec.transpose()*phi).array().pow(2);
		// cout << "tmp:\n "<< tmp << endl;
		VectorXd tmp2 = tmp.row(0);
		MatrixXd tmp_mat = tmp2.asDiagonal();
		// cout << "tmp_mat:\n" << tmp_mat << endl;

		tmp_mat = phi * tmp_mat;
		VectorXd vector_term1 = 3 * tmp_mat.rowwise().sum();
		VectorXd vector_term2 = -3 * The_second_term(Data_a_g, Data_b_g, Data_c_g, Data_a_mu, Data_b_mu, Data_c_mu, one_eigenvec);

		myvector.col(index_k) = vector_term1 + vector_term2;
	}
	New_Phi = phi - myvector*LEARNRATE;//

	return New_Phi;
}

VectorXd The_second_term(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, VectorXd phi)
{
	// phi is a VectorXd
	double para0 = (alpha0 + 1)*(alpha0 + 2) / 2;
	double para1 = alpha0*alpha0;
	double para2 = -alpha0 *(alpha0 + 1) / 2;

	VectorXd Term1 = para0*(phi.dot(Data_a_g))*(phi.dot(Data_b_g)) * Data_c_g;
	VectorXd Term2 = para1*(phi.dot(Data_a_mu))*(phi.dot(Data_b_mu))*Data_c_mu;
	VectorXd Term31 = para2*(phi.dot(Data_a_g))*(phi.dot(Data_b_g))*Data_c_mu;
	VectorXd Term32 = para2*(phi.dot(Data_a_g))*(phi.dot(Data_b_mu))*Data_c_g;
	VectorXd Term33 = para2*(phi.dot(Data_a_mu))*(phi.dot(Data_b_g))*Data_c_g;
	VectorXd output = Term1 + Term2 + Term31 + Term32 + Term33;

	return output;
}



//////////////
Eigen::SparseMatrix<double> edgepot_observables(Node * x1, Node * x2)// x2 is the parent
{
//	cout << "start edgepot_observables function" << endl;
	Eigen::SparseMatrix<double> E_mat;
	E_mat.resize(x1->readsamples().cols(), x2->readsamples().cols());
	E_mat = x1->readsamples().transpose()*x2->readsamples();
	E_mat.makeCompressed(); E_mat.prune(TOLERANCE);


	pair< pair<SparseMatrix<double>, SparseMatrix<double> >, Eigen::SparseVector<double> > U_L = k_svd_observabe(E_mat);
	E_mat.resize(0, 0);
	// p(x1 |h) = U_L.first.first;
	// p(x2 |h) = U_L.first.second;
	// p(h) = U_L.second;
	Eigen::SparseMatrix<double> p_x1_h, p_x2_h, p_x1_h_joint, p_x2_h_joint, p_h_x2, p_x1_x2, p_x1_x2_joint;
	Eigen::SparseVector<double> p_h, p_x2, p_x1;

	p_x1_h.resize(VOCA_SIZE, KHID); p_x1_h = normProbMatrix(U_L.first.first);
	p_x2_h.resize(VOCA_SIZE, KHID); p_x2_h = normProbMatrix(U_L.first.second);
	p_h.resize(KHID); p_h = normProbVector(U_L.second);

	p_x1_h_joint.resize(VOCA_SIZE, KHID);
	p_x2_h_joint.resize(VOCA_SIZE, KHID);

	p_h_x2.resize(KHID, VOCA_SIZE);

	p_x1_x2.resize(VOCA_SIZE, VOCA_SIZE);
	p_x1_x2_joint.resize(VOCA_SIZE, VOCA_SIZE);

	// first get p_x1 and p_x2 :(1) get p_x1_h_joint; (2) get p_x2_h_joint
	p_x1_h_joint = Condi2Joint(p_x1_h, p_h);
	// p_x1_h_joint.colwise().sum() to get p_x1;
	for (int id_row = 0; id_row < p_x1_h_joint.cols(); id_row++)
	{
		Eigen::SparseVector<double> tmp_vec = p_x1_h_joint.block(id_row, 0, 1, p_x1_h_joint.cols());
		p_x1.coeffRef(id_row) = tmp_vec.sum();
	}
	x1->set(p_x1);
	p_x1.resize(0, 0); p_x1_h_joint.resize(0, 0);
	//
	p_x2_h_joint = Condi2Joint(p_x2_h, p_h);
	for (int id_row = 0; id_row < p_x2_h_joint.cols(); id_row++)
	{
		Eigen::SparseVector<double> tmp_vec = p_x2_h_joint.block(id_row, 0, 1, p_x2_h_joint.cols());
		p_x2.coeffRef(id_row) = tmp_vec.sum();
	}
	x2->set(p_x2); p_x2_h_joint.resize(0, 0);
	//p_x1|p_x2
	p_x1_x2 = p_x1_h * p_h_x2;

	p_x1_x2_joint = Condi2Joint(p_x1_x2, p_x2);

//	cout << "end edgepot_observables function" << endl;
#ifdef joint_prob
	//  cout << "p_x1_x2_joint" << p_x1_x2_joint << endl;
	return p_x1_x2_joint;
#endif

#ifdef condi_prob
	//  cout << "p_x1_x2:\n" << p_x1_x2 << endl;
	return p_x1_x2;
#endif

}