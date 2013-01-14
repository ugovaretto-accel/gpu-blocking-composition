

template< typename FunT, template <typename> class KernelT >
void launch(const FunT& f) {

  //KernelT/*<<<1,1>>>*/<FunT>(f);

}


template < typename FunT >
struct k {
/*__global__*/ void v( const FunT& f ) {
  f();
}
};


struct F {
  /*__device__*/ void operator()() const {}
};

int main(int, char**) {
  
  const F ff;
  launch<F,k>(ff);

}
