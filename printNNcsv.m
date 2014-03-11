%% Print the trained neural network parameters to a CSV file
%
load EulerNN
folder = @(name) ['EulerNNcsv/' name '.txt'];

% writeDim = @(name, x) fprintf(fopen(folder(name), 'w'), '%d,%d\n', size(x,1), size(x,2));

param_names = {'Theta1', 'Theta2', 'pca_U', 'pca_mu', 'pca_sigma', 'set_codes', 'RESIZE_DIM'};

for i = 1:numel(param_names)
    par_name = param_names{i};
    fprintf('Writing %s to csv ...\n', par_name);
    par = evalin('base', par_name);
    dlmwrite(folder(par_name), par);
end

clear folder par_name par i

fprintf('DONE!\n');
