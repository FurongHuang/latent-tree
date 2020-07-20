#include "Graph.h"


int main(int argc, const char * argv[]){

      int nProcessors = omp_get_max_threads();
      int iter = 1000;
    cout << "nProcessors:"<< nProcessors<<endl;
    cout << "LEN orig: "<< iter<<endl;
    cout << "LEN revi: "<<  iter-(iter% nProcessors)<<endl;
    cout << "var_1" << "\t" << "var_2" << "\t" << "nonZeros()" << endl;
    omp_set_num_threads(nProcessors);
#pragma omp parallel for
      for (int i=0; i< iter-(iter% nProcessors); i++){
	cout << "There are "<<omp_get_num_threads()<<" threads"<<endl;
      }
}
