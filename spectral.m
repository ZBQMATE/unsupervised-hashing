%spectral hashing
function TRS = spectral(train_data, num_bits)
	
	% train_data, a matrix of [number of batches, number of features];
	% num_bits, int of number of hash code bits
	
	[num_cases case_dim] = size(train_data);
	
	%**********************STEP 1, PCA*************************
	
	num_eigs = min(num_bits, case_dim);
	covariance_data = cov(train_data);
	[vlu, dia] = eigs(covariance_data, num_eigs);
	
	% should be y = vlu * (train_data .- mean(train_data)) ?
	y = train_data * vlu;
	
	%********STEP 2, compute eigen values********
	
	% range is a vector with num_bits length
	range = max(y) - min(y);
	% max_mode <num_bits * 1>
	max_mode = ceil((num_bits + 1) * range / max(range));
	
	num_mode = sum(max_mode) - length(max_mode) + 1;
	
	modes = ones([num_mode num_eigs]);
	m = 1;
	
	for i = 1 : num_eigs
		modes(m + 1 : m + max_mode(i) - 1, i) = 2 : max_mode(i);
		m = m + max_mode(i) - 1;
	end
	
	modes = modes - 1;
	
	omega0 = pi ./ range;
	omegas = modes .* repmat(omega0, [num_mode 1]);
	
	eigval = -sum(omegas .^ 2, 2);
	[yyy, iii] = sort(-eigval);
	
	modes = modes(iii(2 : num_bits + 1), :);
	
	
	%***SAVE***
	
	%spectral_para.pac_vlu = vlu;
	%spectral_para.data_max = max(y);
	%spectral_para.data_min = min(y);
	%spectral_para.modes = modes;
	
	% ***********compress*********
	
	y = y - repmat(min(y), [num_cases 1]);
	
	omegas = modes .* repmat(omega0, [num_bits 1]);
	
	% TRS is num_cases * num_bits
	TRS = zeros([num_cases num_bits]);
	
	for i = 1 : num_bits
		omegai = repmat(omegas(i, :), [num_cases 1]);
		ys = sin(y .* omegai + pi / 2);
		yi = prod(ys, 2);
		TRS(:, i) = yi;
	end