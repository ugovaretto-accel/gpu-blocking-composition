

template< typename FunT, template <typename> class KernelT >
void launch(FunT f) {

  KernelT/*<<<1,1>>>*/<FunT>(f);

}


template < typename FunT >
/*__global__*/ void k( FunT f ) {
  f();
}


struct F {
  /*__device__*/ void operator()() const {}
};

int main(int, char**) {

  launch<F,k>(F());

}
