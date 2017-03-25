%pca ordinary hashing
function TRS = pcahash(train_data, num_bits)
	
	% train_data, a matrix of [number of cases, number of features];
	%num_bits, int number of hash code bits
	
	[num_cases case_dim] = size(train_data);
	
	%PCA hashing
	
	num_eigs = min(num_bits, case_dim);
	covariance_data = cov(train_data);
	
	[vlu, dia] = eigs(covariance_data, num_eigs);
	
	y = (train_data - mean(mean(train_data))) * vlu;
	
	TRS = y;