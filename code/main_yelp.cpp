//
//  main.cpp
//  latenttree
//
//  Created by Furong Huang on 11/28/13.
//  Copyright (c) 2013 Furong Huang. All rights reserved.
//

#include "Graph.h"

typedef unsigned long long timestamp_t;
typedef int bool_t;
// timeval
timeval start_timeval_readfile, stop_timeval_readfile;
timeval start_timeval_mst_graph, stop_timeval_mst_graph;
timeval start_timeval_svd_dis, stop_timeval_svd_dis;
timeval start_timeval_mst, stop_timeval_mst;
timeval start_timeval_rg, stop_timeval_rg;
timeval start_timeval_merge, stop_timeval_merge;
//timeval start_timeval_RF, stop_timeval_RF;
//timestamp_t
timestamp_t measure_start_readfile, measure_stop_readfile;       // Timing for svd1
timestamp_t measure_start_mst_graph, measure_stop_mst_graph;       // Timing for svd2
timestamp_t measure_start_svd_dis, measure_stop_svd_dis;       // Timing for reading before pre_proc
timestamp_t measure_start_mst, measure_stop_mst;       // Timing for reading after stpm
timestamp_t measure_start_rg, measure_stop_rg;       // Timing for pre_proc
timestamp_t measure_start_merge, measure_stop_merge;     // Timing for stpm
//timestamp_t measure_start_RF, measure_stop_RF;     // Timing for error_calc
// time
double time_readfile, time_mst_graph,  time_svd_dis, time_mst, time_rg, time_merge;                 // Time taken

using namespace std;
using namespace Eigen;
int main(int argc, const char * argv[])
{
    
    // insert code here...
    
    
    
    
    
    gettimeofday(&start_timeval_readfile, NULL);
    // Furong: generate nodes here. Using node constructor
    std::vector<int> umark;         //setmark
    vector<Node> mynodelist;
    for(int i=0;i<NVAR;i++)
    {
        Node tmpnode(i,i,'0');// constructor for node: whose id is i and whose mark is i.
        mynodelist.push_back(tmpnode);
        umark.push_back(i);
    }
    //////////////////////////////////////////////////////////////////
    // read the adjacency submatrices from the dataset
    vector<Node> *mynodelist_ptr = & mynodelist;
    read_G_vec((char *)FILE_G, mynodelist_ptr);
    
    gettimeofday(&stop_timeval_readfile, NULL);
    measure_stop_readfile = stop_timeval_readfile.tv_usec + (timestamp_t)stop_timeval_readfile.tv_sec * 1000000;
    measure_start_readfile = start_timeval_readfile.tv_usec + (timestamp_t)start_timeval_readfile.tv_sec * 1000000;
    time_readfile = (measure_stop_readfile - measure_start_readfile) / 1000000.0L;
    printf("Exec Time reading matrices before preproc = %5.25e (Seconds)\n",time_readfile);
    
    
    gettimeofday(&start_timeval_svd_dis, NULL);
    
    Eigen::MatrixXd adjMatrix(NVAR,NVAR);
    Eigen::SparseMatrix<double> adjMatrix_sparse;
    adjMatrix_sparse.resize(NVAR,NVAR);
    
#ifdef useDistance
    
    adjMatrix = read_G_dense(adjMatrix_name,"adj_G_name", NVAR,NVAR);
    cout <<"adjMatrix.coeff(0,0):"<<adjMatrix.coeff(0,0)<<endl;
    cout <<"adjMatrix.coeff(0,1):"<<adjMatrix.coeff(0,1)<<endl;
    
#endif
    
#ifdef nouseDistance
    
    //     Eigen::SparseMatrix<double> adjMatrix;
    //     adjMatrix.resize(NVAR, NVAR);
    //     adjMatrix.reserve(NVAR*NVAR);
    
    // generate adjMatrix
    cout << "=======Begin distance calculation!=======" << endl;
    vector<int> iter_row;
    vector<int> iter_col;
    vector<int> num_coreviews;
    for (int row_id =0; row_id < NVAR; row_id++)
    {
        for (int col_id = row_id+1; col_id < NVAR; col_id++)
        {
            iter_row.push_back(row_id);
            iter_col.push_back(col_id);
        }
    }
    int nProcessors = omp_get_max_threads();
    cout << "nProcessors:"<< nProcessors<<endl;
    cout << "LEN orig: "<< iter_row.size()<<endl;
    cout << "LEN revi: "<<  iter_row.size()-(iter_row.size()% nProcessors)<<endl;
    cout << "var_1" << "\t" << "var_2" << "\t" << "nonZeros()" << endl;
    omp_set_num_threads(nProcessors-12);
    //#pragma omp parallel for 
    for (int i=0; i< iter_row.size()-(iter_row.size()% nProcessors); i++){
	//	cout << "There are "<<omp_get_num_threads()<<" threads"<<endl;
      cout << "start"<< endl;
      int row_id = iter_row[i];
      int col_id = iter_col[i];
      //       	cout  << row_id << "\t "<< col_id << "\t";
      //      #pragma omp critical
      pair<double, int>prod_pair = prod_sigvals(&(mynodelist[row_id]), &(mynodelist[col_id]));
      //	cout << "success prod_sigvals"<< endl;
      double value_prod = prod_pair.first;
      
      //#pragma omp critical  
      {
	cout << row_id << "\t" << col_id << endl;
	num_coreviews.push_back(prod_pair.second);
	if (value_prod < EPS)
	  {
	    adjMatrix.coeffRef(row_id,col_id)=MAXDISTANCE;
	    adjMatrix.coeffRef(col_id,row_id)=MAXDISTANCE;
	  }
	else
	  {
	    adjMatrix.coeffRef(row_id,col_id)=-log(value_prod);
	    adjMatrix.coeffRef(col_id,row_id)=adjMatrix.coeff(row_id,col_id);
	  }
      }
    }
    cout << endl;
    cout << "end of fist openmp for loop"<< endl;
    //#pragma omp parallel for
    for (int i=  iter_row.size()-(iter_row.size()% nProcessors); i< iter_row.size(); i++){
      int row_id = iter_row[i];
      int col_id = iter_col[i];
      //      cout  << row_id << "\t "<< col_id << "\t";
      pair<double, int>prod_pair = prod_sigvals(&(mynodelist[row_id]), &(mynodelist[col_id]));
      double value_prod = prod_pair.first;
      
      //      #pragma omp critical
      {
	num_coreviews.push_back(prod_pair.second);
	if (value_prod < EPS)
	  {
	    adjMatrix.coeffRef(row_id,col_id)=MAXDISTANCE;
	    adjMatrix.coeffRef(col_id,row_id)=MAXDISTANCE;
	  }
	else
	  {
	    adjMatrix.coeffRef(row_id,col_id)=-log(value_prod);
	    adjMatrix.coeffRef(col_id,row_id)=adjMatrix.coeff(row_id,col_id);
	  }
      }
    }
    
    

      adjMatrix_sparse = adjMatrix.sparseView();
      write_vector("result/NUM_co-reviews.TXT", num_coreviews);
      write_sparseMat("result/DistanceMatrix.TXT", adjMatrix_sparse);
#endif
      adjMatrix_sparse = adjMatrix.sparseView();
      gettimeofday(&stop_timeval_svd_dis, NULL);
      measure_stop_svd_dis = stop_timeval_svd_dis.tv_usec + (timestamp_t)stop_timeval_svd_dis.tv_sec * 1000000;
      measure_start_svd_dis = start_timeval_svd_dis.tv_usec + (timestamp_t)start_timeval_svd_dis.tv_sec * 1000000;
      time_svd_dis = (measure_stop_svd_dis - measure_start_svd_dis) / 1000000.0L;
      printf("Exec Time SVD distance  = %5.25e (Seconds)\n",time_svd_dis);
      
      
      int c = 0;
      std::vector<Edge> EV;    //All edges
    
      long long int NUM_EDGES = NVAR*(NVAR-1);
      for(int i = 0; i < NVAR; i++) {         //All edges have cycles,
        for(int j = 0; j < i; j++) {
	  if(i == j) continue;
	  if(adjMatrix(i,j) > 0 && c < NUM_EDGES) {
	    Edge new_edge(&mynodelist[i], &mynodelist[j], adjMatrix(i,j));
	    EV.push_back(new_edge);
	    c++;                    //c = NUM_EDGES after loop
	  }
        }
      }
      cout <<"==================8888888888888888============================"<< endl;
      cout << "All Vertices: #" << NVAR << endl; cout << "All Edges: #" << c << endl;
      
      
      //////////////////////
      gettimeofday(&start_timeval_mst, NULL);
      
      std::vector<Edge *> mst;   //mst
      cout << "-----------beginning of kruskal-----------------" << endl;
      //call Kruskal function
      std::sort(EV.begin(),EV.end(),compareByWeightObj);
      mst = KruskalMST(EV, mynodelist);
      
      
      gettimeofday(&stop_timeval_mst, NULL);
      measure_stop_mst = stop_timeval_mst.tv_usec + (timestamp_t)stop_timeval_mst.tv_sec * 1000000;
      measure_start_mst = start_timeval_mst.tv_usec + (timestamp_t)start_timeval_mst.tv_sec * 1000000;
      time_mst = (measure_stop_mst - measure_start_mst) / 1000000.0L;
      printf("Exec Time MST  = %5.25e (Seconds)\n",time_mst);
      
      
      /////////////////////////////////////////////////////////////////
      //Build a MST graph
      gettimeofday(&start_timeval_mst_graph, NULL);
      cout << "------End of Kruskal: starting to build g_mst----"<< endl;
      Graph g_mst(mst);
      
      
      //split neighborhood to each RG algorithm
      vector<int> internal_nodes;
      for (int num = 0; num < g_mst.readadj().size(); num++)
	{
	  if (g_mst.readadj()[num].size()>1)
	    {
	      internal_nodes.push_back(num);
	      //  cout << "Internal Node :" << num << endl;
	    }
	}
      cout << "internal_nodes.size():  "<< internal_nodes.size() << endl;
      
      
      gettimeofday(&stop_timeval_mst_graph, NULL);
      measure_stop_mst_graph = stop_timeval_mst_graph.tv_usec + (timestamp_t)stop_timeval_mst_graph.tv_sec * 1000000;
      measure_start_mst_graph = start_timeval_mst_graph.tv_usec + (timestamp_t)start_timeval_mst_graph.tv_sec * 1000000;
      time_mst_graph= (measure_stop_mst_graph - measure_start_mst_graph) / 1000000.0L;
      printf("Exec Time MST graph  = %5.25e (Seconds)\n",time_mst_graph);
      
      // write the mst to file
      // cout << "====================MST results===================="<<endl;
      // g_mst.displayadj_edgeD();
      // cout << "====================END of MST results===================="<<endl;
      
      
      gettimeofday(&start_timeval_rg, NULL);
      cout << "=======================RG starts=========================="<< endl;
      vector<Graph *> g_RG;
      for (int ind_rg = 0; ind_rg < internal_nodes.size(); ind_rg++)
	{
	  cout << "ind_rg: " << ind_rg << endl;
        //     cout <<"start to build new neighborhood graph"<<endl;
        Graph * g_rg = new Graph(g_mst.readadj()[internal_nodes[ind_rg]], &mynodelist[internal_nodes[ind_rg]]);
        RG(g_rg, &adjMatrix_sparse);
        g_RG.push_back(g_rg);
        int num_nodes_thisRG_x = g_rg->readnum_N();
        int num_nodes_thisRG_h = g_rg->readnum_H();
        
    }
    adjMatrix.resize(0,0);
    
    gettimeofday(&stop_timeval_rg, NULL);
    measure_stop_rg = stop_timeval_rg.tv_usec + (timestamp_t)stop_timeval_rg.tv_sec * 1000000;
    measure_start_rg = start_timeval_rg.tv_usec + (timestamp_t)start_timeval_rg.tv_sec * 1000000;
    time_rg= (measure_stop_rg - measure_start_rg) / 1000000.0L;
    printf("Exec Time RG  = %5.25e (Seconds)\n",time_rg);
    
    
    
    
    // merging step
    
    gettimeofday(&start_timeval_merge, NULL);
    cout<<"===start merging step!!!===="<<endl;
    Graph * g_Merged = g_RG[0];
    g_RG.erase(g_RG.begin());
    
    int curr_id = 0;
    while(g_RG.size()>0)
    {
        bool merged = Graph_merge(g_Merged,g_RG[curr_id]);
        if (merged==false)
        {
            curr_id++;
        }
        else{
            g_RG.erase(g_RG.begin()+curr_id);// check this!
            curr_id = 0;
        }
        
    }
    cout << "The final RG forest: Number of observable nodes: "<<  g_Merged->readnum_N() << endl;
    cout << "The final RG forest: Number of hidden nodes: " <<  g_Merged->readnum_H()<< endl;
    
    SparseMatrix<double> category;
    category=g_Merged->estimateCategory();
    write_sparseMat("result/category.TXT", category);
    
    gettimeofday(&stop_timeval_merge, NULL);
    measure_stop_merge = stop_timeval_merge.tv_usec + (timestamp_t)stop_timeval_merge.tv_sec * 1000000;
    measure_start_merge = start_timeval_merge.tv_usec + (timestamp_t)start_timeval_merge.tv_sec * 1000000;
    time_merge= (measure_stop_merge - measure_start_merge) / 1000000.0L;
    printf("Exec Time MERGE  = %5.25e (Seconds)\n",time_merge);
    
    
    g_Merged->displayadj_edgeD("result/nodelist.TXT","result/adjsparse.TXT","result/neighborhood.TXT","result/edgePot.TXT");
    //      cout <<  "====================END of RG results===================="<<endl;
    cout << "================================="<< endl;
    printf("EXEC TIME Summary: \n");
    printf("Time reading matrices   = %5.25e (Secs)\n",time_readfile);
    printf("Exec Time SVD= %5.25e (Secs), MST =%5.25e (Secs), MSTgraph = = %5.25e (Secs), RG = %5.25e (Secs), Merge = %5.25e (Secs)\n",time_svd_dis,time_mst, time_mst_graph,time_rg,time_merge);
    printf("Total EXEC time = %5.25e (Secs)\n ", time_svd_dis+time_mst+time_mst_graph+time_rg+time_merge);
    
    
    return 0;
}

