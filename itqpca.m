%iterative quantization pca
function TRS = itqpca(train_data, num_bits)
	
	%train_data, a matrix of [number of cases, number of features];
	%num_bits, int number of hash code bits
	
	[num_cases case_dim] = size(train_data);
	
	%**** step 1, PCA *****
	
	num_eigs = min(num_bits, case_dim);
	covariance_data = cov(train_data);
	
	[vlu, dia] = eigs(covariance_data, num_eigs);
	
	%****** step 2, ITQ ******
	
	% x <num_cases * case_dim>
	% w <case_dim * num_bits>
	% v, bin, vr <num_cases * num_bits>
	% r <num_bits * num_bits>
	
	
	% v = x * w
	v = (train_data - mean(mean(train_data))) * vlu;
	
	%bin = sgn(x * w * r)
	
	%ini r
	[r, sjtu, zju] = svd(randn(num_bits, num_bits));
	
	for iters = 1 : 50
		
		%fix vr
		vr = v * r;
		bin = vr > 0;
		
		%orthogonal procrustes analysis
		[a, bibbm, b] = svd(bin' * v);
		r = b * a';
		
	end
	
	TRS = vr;