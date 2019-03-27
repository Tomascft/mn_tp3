#include <stdio.h>
#include <omp.h>

#include <smmintrin.h>
#include <x86intrin.h>

#define NBEXPERIMENTS    22
static long long unsigned int experiments [NBEXPERIMENTS] ;


#define N            4096             
#define TILE           16

typedef double vector [N] __attribute__ ((aligned (16))) ;

static vector a, b ;

static const float duree_cycle = (float) 1 / (float) 2.6;	// duree du cycle en nano seconde 10^-9

void calcul_flop (char *message, int nb_operations_flottantes,
	     unsigned long long int cycles)
{
  printf ("%s %d operations %f GFLOP/s\n\n", message, nb_operations_flottantes,
	  ((float) nb_operations_flottantes) / (((float) cycles) * duree_cycle));
  return;
}

long long unsigned int average (long long unsigned int *exps)
{
  unsigned int i ;
  long long unsigned int s = 0 ;

  for (i = 2; i < (NBEXPERIMENTS-2); i++)
    {
      s = s + exps [i] ;
    }

  return s / (NBEXPERIMENTS-2) ;
}


void init_vector (vector X, const double val)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    X [i] = val ;

  return ;
}

void print_vectors (vector X, vector Y)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    printf (" X [%d] = %le Y [%d] = %le\n", i, X[i], i,Y [i]) ;

  return ;
}

double dot0 (vector X, vector Y)
{
  register unsigned int i ;
  register double dot ;

  dot = 0.0 ;

  for (i=0; i < N; i++)
    dot += X [i] * Y [i];

  return dot ;
}

double dot1 (vector X, vector Y)
{
  register unsigned int i ;
  register double dot ;
  
  dot = 0.0 ;
#pragma omp parallel for schedule(static) reduction (+:dot)
  for (i=0; i < N; i++)
    dot += X [i] * Y [i];

  return dot ;
}

double dot2 (vector X, vector Y)
{
  register unsigned int i ;
  register double dot ;

  dot = 0.0 ;
#pragma omp parallel for schedule(dynamic) reduction (+:dot)
  for (i=0; i < N; i++)
    dot += X [i] * Y [i];

  return dot ;
}

double dot3 (vector X, vector Y)
{
  register unsigned int i ;
  register double dot ;

  dot = 0.0 ;
#pragma omp parallel for schedule(static) reduction (+:dot)
  for (i = 0; i < N; i = i + 4)
    {
    dot += X [i] * Y [i];
    dot += X [i + 1] * Y [i + 1];
    dot += X [i + 2] * Y [i + 2];
    dot += X [i + 3] * Y [i + 3];
    }

  return dot ;
}

double dot4 (vector X, vector Y)
{
  __m128d v1, v2, res ;
  register unsigned int i ;
  double dot [2] __attribute__ ((aligned (16))) ;
  double dot_total = 0.0 ;

    for (i = 0; i < N; i = i + 2)
    {
      v1 = _mm_load_pd (X+i) ;
      v2 = _mm_load_pd (Y+i) ;

      res = _mm_dp_pd (v1, v2, 0xFF) ;

      _mm_store_pd (dot, res) ;

      dot_total += dot [0] ;
    }

    return dot_total ;
  
}

int main ()  
{
  unsigned long long int start, end ;

  unsigned long long int av ;
  
  double r ;

  int exp ;
  
  /* 
     rdtsc: read the cycle counter 
  */
  
  printf ("====================DOT =====================================\n") ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;
  
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         r = dot0 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("dot0 (one core) : r = %f \t\t\t %Ld cycles\n", r, av) ;
  calcul_flop ("dot0 (one core) ", 2*N, av) ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;
  
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         r = dot4 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("dot4 (one core vectorization) : r = %f \t %Ld cycles\n", r, av) ;
  calcul_flop ( "dot4 (one core vectorization) ", 2*N, av) ;
    
  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;
  
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         r = dot1 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("dot1 r = %f OpenMP static loop:\t\t %Ld cycles\n", r, av) ;
  calcul_flop ( "dot1 (OpenMP static loop) ", 2*N, av) ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         r = dot2 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  
  printf ("dot2 r = %f OpenMP dynamic loop:\t\t %Ld cycles\n", r, av) ;
  calcul_flop ( "dot2 (OpenMP dynamic loop) " , 2*N, av) ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         r = dot3 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  
  printf ("dot3 r = %f OpenMP static unrolled loop:\t %Ld cycles\n", r, av) ;
  calcul_flop ("dot3 (OpenMP static unrolled loop) ", 2*N, av) ;
  
  printf ("=============================================================\n") ;

}



