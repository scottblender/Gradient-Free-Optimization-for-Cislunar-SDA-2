function meas_noise = generate_measurement_noise(num_steps, num_obs, R, seed)
    if nargin >= 4 && ~isempty(seed)
        rng(seed, 'twister');
    end

    meas_noise = zeros(num_steps, num_obs, 2);
    for k = 2:num_steps
        for i = 1:num_obs
            meas_noise(k,i,:) = mvnrnd([0;0], R, 1);
        end
    end
end