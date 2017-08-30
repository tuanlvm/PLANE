# PLANE
PLANE: Probabilistic Latent Document Network Embedding

INTRODUCTION

This is an implementation of PLANE - a method for embedding a document network in a low dimensional space from Le & Lauw (ICDM 2014).

Usage:

	perl plane.pl		--num_topics $num_topics
				--dim $dim
				--alpha $alpha
				--beta $beta
				--gamma $gamma
				--EM_iter $EM_iter
				--Quasi_iter $Quasi_iter
				--data $data
				--graph $graph
				--output_file $output_file

Arguments:

	$num_topics: number of topics
	$dim: number of dimensions (default 2)
	$alpha: Dirichlet parameter (default 0.01)
	$beta: covariance for Gaussian prior of topic coordinates (default 0.1*$num_docs)
	$gamma: covariance for Gaussian prior of document coordinates (default 0.1*$num_topics)
	$EM_iter: number of iterations for EM (default 100)
	$Quasi_iter: maximum iterations of Quasi-Newton (default 10)
	$data: input data
	$graph: document network
	$output_file: output file

Details:

+ This implementation needs Algorithm::LBFGS library for quasi-Newton method L-BFGS.
  The library can be downloaded at http://search.cpan.org/~laye/Algorithm-LBFGS-0.16/lib/Algorithm/LBFGS.pm.
  To install,
	
	  cpan Algorithm::LBFGS
	
+ Example of input data with 3 documents (numbers are ids of words):

	0 1 1 2 2 3 4 4 5 6 7 7 8 8 8 8 9 10 11 12 13 13 14 14 15 15 15 16<br/>
	17 18 19 20 20 21 22 23 24 25 25 25 25 25 25 26 27 27 28 29<br/>
	30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 50 51 52 53 54 54 55 56 57 58<br/>
			
+ Document network is represented by a matrix A: NxN. N is the number of documents and A[i,j]=A[j,i]=1 when there is an edge connecting documents i and j.

  For example,
  
	0 1 0<br/>
	1 0 1<br/> 
	0 1 0

HOW TO CITE

If you use PLANE for your research, please cite:

	@inproceedings{plane,
	    title={Probabilistic Latent Document Network Embedding},
	    author={Le, Tuan MV and Lauw, Hady W},
	    booktitle={IEEE International Conference on Data Mining},
	    year={2014}
	}
		
The paper can be downloaded from: http://www.hadylauw.com/publications/icdm14.pdf
