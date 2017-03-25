%anchor graph hashing
function TRS = anchorgraph(train_data, num_bits)
	
	% train_data, a matrix of [number of cases, number of fetures]
	% num_bits, int number of hash code bits
	
	
	%param
	num_anchors = 50 ;
	num_nearest_anchor = 5;
	
	[num_cases case_dim] = size(train_data);
	
	% anchor graph hashing
	
	%compute the anchor, using k mean algo
	
	anchor = rand([num_anchors case_dim]);
	
	% ******k mean********
	databox = ones([num_anchors case_dim (num_cases/num_anchors)]);
	temp = 5 * ones([1 case_dim]);
	rst_data = train_data;
	
	for iters = 1 : 50
	
		for p = 1 : (num_cases/num_anchors)
			for i = 1 : num_anchors
				for j = 1 : size(rst_data,1)
				
					gap = anchor(i, :) - rst_data(j, :);
					dis_pow = sum(gap .^ 2);
					
					if dis_pow < sum((anchor(i, :) - temp) .^ 2)
						temp = rst_data(j, :);
						temp_idx = j;
					end
					
				end
				
				%save temp
				databox(i, :, p) = temp;
				temp = 5 * ones([1 case_dim]);
				rst_data(temp_idx, :) = [];
				
			end
		end
		
		% calculate the mean and update anchor
		for i = 1 : num_anchors
			anchor(i, :) = mean(databox(i,:,:), 3);
			%<1*case_dim> = mean(<1*case_dim*num_cases/num_anchors>,3)
		end
		
		rst_data = train_data;
	end
	
	%*****anchor graph*****
	
	z = zeros([num_cases num_anchors]);
	
	train_data_pow = sum(train_data .^ 2, 2);
	anchor_pow = sum(anchor .^ 2, 2);
	train_data_anchor = train_data * anchor';
	
	dis = abs(repmat(train_data_pow, [1 size(anchor_pow', 2)]) + repmat(anchor_pow', [size(train_data_pow', 2) 1]) - 2 * train_data_anchor);
	
	val = zeros([num_cases num_nearest_anchor]);
	pos = val;
	
	for i = 1 : num_nearest_anchor
	
		[val(:, i), pos(:, i)] = min(dis, [], 2);
		tep = (pos(:, i) - 1) * num_cases + [1 : num_cases]';
		dis(tep) = 1e60;
		
	end
	
	sigma = mean(val(:, num_nearest_anchor) .^ 0.5);
	
	val = exp(-val / (1/1*sigma^2));
	val = repmat(sum(val,2) .^ (-1), 1, num_nearest_anchor) .* val;
	
	tep = (pos - 1) * num_cases + repmat([1 : num_cases]', 1, num_nearest_anchor);
	
	z([tep]) = [val];
	z = sparse(z);
	
	%compute eigen
	
	lamda = sum(z);
	m = z'*z;
	m = diag(lamda .^ (-0.5)) * m * diag(lamda .^ (-0.5));
	
	[w, v] = eig(full(m));
	
	eigenvalue = diag(v)';
	
	[eigenvalue, order] = sort(eigenvalue, 'descend');
	
	w = w(:, order);
	
	ind = find(eigenvalue > 0 & eigenvalue < 1 - 1e-3);
	eigenvalue = eigenvalue(ind);
	
	w = w(:, ind);
	w = diag(lamda .^ (-0.5)) * w(:, 1 : num_bits) * diag(eigenvalue(1 : num_bits) .^ (-0.5));
	
	TRS = z * w;