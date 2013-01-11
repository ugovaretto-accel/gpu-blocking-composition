
/*
Copyright (c) 2012, MAURO BIANCO, UGO VARETTO, SWISS NATIONAL SUPERCOMPUTING CENTRE (CSCS)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Swiss National Supercomputing Centre (CSCS) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL MAURO BIANCO, UGO VARETTO, OR 
SWISS NATIONAL SUPERCOMPUTING CENTRE (CSCS), BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <GSCL/GSCL.h>
#include "GSCL/3D/stencil_lib_3x3x3.h"
#include "GSCL/3D/stencil_lib_1x1x1.h"
#include "GSCL/iteration_space.h"
#include "GSCL/3D/Grid.h"
#include "GSCL/defaults.h"
#include "utils/Timer.h"

#include <string> 

/** Program created to develop MPI implementation
 */

struct uxx1 {
  typedef boost::fusion::vector<GSCL::flag_read,GSCL::flag_write,GSCL::flag_read,GSCL::flag_read,GSCL::flag_read> access_list_type;

  const double dt, dh;

  uxx1(double const _dt, double const _dh): dt(_dt), dh(_dh) {}

  template <typename td1,typename tu1,typename txx,typename txy,typename txz>
  void operator()(td1 const& d1, 
                  tu1 & u1, 
                  txx const& xx, 
                  txy const& xy, 
                  txz const& xz) const {
    const double dth = dt/dh;
    const double c1 = 9./8.;
    const double c2 = -1./24.;

    const double d = 0.25*(d1(0,0,0)+d1(0,-1,0)+
                           d1(0,0,-1)+d1(0,-1,-1));
     
      u1()=u1(0,0,0)+(dth/d)*(
                              c1*(xx(0,0,0)-xx(-1,0,0))+ 
                              c2*(xx(1,0,0)-xx(-2,0,0))+
                              c1*(xy(0,0,0)-xy(0,-1,0))+
                              c2*(xy(0,1,0)-xy(0,-2,0))+
                              c1*(xz(0,0,0)-xz(0,0,-1))+
                              c2*(xz(0,0,1)-xz(0,0,-2))
                              );

  }
}; 

MAKE_STENCIL_3D(st_d1, 0, 0, 1, 0, 1, 0)
MAKE_STENCIL_3D(st_xx, 2, 1, 0, 0, 0, 0)
MAKE_STENCIL_3D(st_xy, 0, 0, 2, 1, 0, 0)
MAKE_STENCIL_3D(st_xz, 0, 0, 0, 0, 2, 1)


struct init_f {
  typedef boost::fusion::vector<GSCL::flag_init,GSCL::flag_init,GSCL::flag_init,GSCL::flag_init,GSCL::flag_init> access_list_type;

  int n,m;
  double c;

  explicit init_f(int _n, int _m, double _c=0.01): n(_n), m(_m), c(_c) {}

  template<typename S1,typename S2,typename S3,typename S4,typename S5>
  void operator()(S1& u1,S2& u2,S3& u3,S4& u4,S5& u5) const {
    int i,j,k;
    u1.global_index(i,j,k);
    u1() = c;
    u2() = ((double)(i-j))*c;
    u3() = ((double)i)*c*c;
    u4() = ((double)j)*c;
    u5() = ((double)j/(double)(j+1));
  }
}; 

struct print
  : GSCL::unary_op
{
  typedef boost::fusion::vector<GSCL::flag_read> access_list_type;

  template <typename stencil>
  void operator()(stencil const & u) const 
  {
    int i,j,k;
    u.global_index(i,j,k);

    GSCL::cout << GSCL::GSCL_pid() << ":grep ( " << i << " , " << j << " , " << k << " ) = " << u() << "\n";

  }
};

int main(int argc, char** argv) {

  GSCL::GSCL_Init(argc, argv);

  int result = 0;

  int dim_n_1 = atoi(argv[1]);
  int dim_m_1 = atoi(argv[2]);
  int dim_l_1 = atoi(argv[3]);
  int nrep    = atoi(argv[4]);

  int R,C,S;
  GSCL::_3D_process_grid->dims(R,C,S);

  int dim_n = dim_n_1 * R;
  int dim_m = dim_m_1 * C;
  int dim_l = dim_l_1 * S;

  GSCL::cout << "Weak scaling: " << R << "x" << C << "x" << S
            << " with " << GSCL::GSCL_THREADS << " threads, " 
            << ": size " << dim_n << "x" << dim_m << "x" << dim_l << GSCL::endl;

  //typedef GSCL::make_arch<3, GSCL::mpi, GSCL::openmp, GSCL::sequential>::arch_type arch_type;
  typedef GSCL::default_3D_arch arch_type;

  typedef GSCL::default_SBStorage<arch_type, double>::type storage_type;
  storage_type storage1(dim_n, dim_m, dim_l, 2);
  storage_type storage2(dim_n, dim_m, dim_l, 2);
  storage_type storage3(dim_n, dim_m, dim_l, 2);
  storage_type storage4(dim_n, dim_m, dim_l, 2);
  storage_type storage5(dim_n, dim_m, dim_l, 2);


  GSCL::Grid3D< stencil_1x1x1_stateful,storage_type > igrid1(storage1, dim_n, dim_m, dim_l);
  GSCL::Grid3D< stencil_1x1x1_stateful,storage_type > igrid2(storage2, dim_n, dim_m, dim_l);
  GSCL::Grid3D< stencil_1x1x1_stateful,storage_type > igrid3(storage3, dim_n, dim_m, dim_l);
  GSCL::Grid3D< stencil_1x1x1_stateful,storage_type > igrid4(storage4, dim_n, dim_m, dim_l);
  GSCL::Grid3D< stencil_1x1x1_stateful,storage_type > igrid5(storage5, dim_n, dim_m, dim_l);

  GSCL::Grid3D< st_d1,storage_type > grid1=igrid1.reshape<st_d1>(shrink(igrid1.corespace(),2,2,2,2,2,2));
  GSCL::Grid3D< stencil_1x1x1,storage_type > grid2=igrid2.reshape<stencil_1x1x1>(shrink(igrid2.corespace(),2,2,2,2,2,2));
  GSCL::Grid3D< st_xx,storage_type > grid3=igrid3.reshape<st_xx>(shrink(igrid3.corespace(),2,2,2,2,2,2));
  GSCL::Grid3D< st_xy,storage_type > grid4=igrid4.reshape<st_xy>(shrink(igrid4.corespace(),2,2,2,2,2,2));
  GSCL::Grid3D< st_xz,storage_type > grid5=igrid5.reshape<st_xz>(shrink(igrid5.corespace(),2,2,2,2,2,2));

  GSCL::default_context<arch_type>::type ctx = GSCL::GSCL_Begin<arch_type>();
  //  GSCL::context<GSCL::mpi> ctx = GSCL::GSCL_Begin<GSCL::mpi>();


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////


  Timer t;
  t.Start();
  GSCL::do_all(ctx, igrid1, igrid2, igrid3, igrid4, igrid5, init_f(dim_n, dim_m));
  double time = t.Stop();

  GSCL::cout << "Init Time: " << time << GSCL::endl;
  GSCL::cout << "after init one grid" << GSCL::endl;

  t.Start();
  for (int i = 0; i<nrep; ++i) 
    GSCL::do_all(ctx, grid1, grid2, grid3, grid4, grid5, uxx1(0.01, 0.1));
  time = t.Stop();

  GSCL::cout << "Exec Time: " << time/nrep << GSCL::endl;

  GSCL::GSCL_End(ctx);

  //#define DUMPDATA
#ifdef DUMPDATA
  GSCL::cout <<  "Dumping output" << GSCL::endl;

  typedef GSCL::make_arch<3, GSCL::sequential>::arch_type out_arch_type;
  GSCL::default_context<out_arch_type>::type ctx_seq = GSCL::GSCL_Begin<out_arch_type>();

  for (int i=0; i<GSCL::GSCL_procs(); ++i) {
    if (GSCL::GSCL_pid() == i)
      GSCL::do_all(ctx_seq, igrid2.local_grid(), print());
    MPI_Barrier(GSCL::GSCL_WORLD);
  }

  GSCL::GSCL_End(ctx_seq);
#endif

  storage1.destroy();
  storage2.destroy();
  storage3.destroy();
  storage4.destroy();
  GSCL::GSCL_Finalize();

  return 0;
}
