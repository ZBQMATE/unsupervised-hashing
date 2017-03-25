
function runhash(model, bits)

	% parameters : run hash model; output bit length; 
	
	warning('error', 'Octave:broadcast');
	
	if exist('page_output_immediately')
		page_output_immediately(1); 
	end
	
	more off;
	
	from_data_file = load('data.mat');
	usps_data = from_data_file.data;
	
	% 256 * 9000 testing data, 256 * 1000 training data, 256 * 1000 validation data
	
	train_batch.inputs = usps_data.training.inputs;
	train_batch.targets = usps_data.training.targets;
	
	validation_batch.inputs = usps_data.validation.inputs;
	validation_batch.targets = usps_data.validation.targets;
	
	%train data <1000 * 256>
	train_data = transpose(train_batch.inputs);
	[num_cases case_dim] = size(train_data);
	
	%*************chose model*************
	
	%model 1, spectral hashing
	if model == 1
		% TRS_MTX is num_cases * num_bits
		TRS_MTX = spectral(train_data, bits);
	end
	
	%model 2, PCA hashing
	if model == 2
		% TRS_MTX is num_cases * num_bits
		TRS_MTX = pcahash(train_data, bits);
	end
	
	%model 3, anchor graph hashing
	if model == 3
		% TRS_MTX is num_cases * num_bits
		TRS_MTX = anchorgraph(train_data, bits);
	end
	
	%model 4, ITQ PCA hashing
	if model == 4
		% TRS_MTX is num_cases * num_bits
		TRS_MTX = itqpca(train_data, bits);
	end
	
	
	%binary hash code, num_cases * num_bits
	BIN_HASH = TRS_MTX > 0;
	
	%check the result
	temp = zeros([num_cases/10 bits]);
	idx = 1;
	hamming_diatance = 0;
	
	for tgt = 1 : 10
		
		for i = 1 : num_cases
			if train_batch.targets(tgt, i) == 1
				temp(idx, :) = BIN_HASH(i, :);
				idx++;
			end
		end
		
		%calculate the total hamming distance
		for i = 1 : (num_cases/10)
			for j = 1 : (num_cases/10)
				hamming_diatance = hamming_diatance + sum(temp(i, :) - temp(j, :) != 0);
			end
		end
		
		idx = 1;
		temp = zeros([num_cases/10 bits]);
		
	end
	
	fprintf('total hamming distance is %d', hamming_diatance);