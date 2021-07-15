
%%
% Code is based on https://github.com/pdollar
toolbox_path = fullfile('utils', 'toolbox');
edges_path = fullfile('utils', 'edges');

addpath('utils')
addpath(edges_path)
addpath(genpath(toolbox_path))

% Suppress Warning regarding image size
warning('off', 'Images:initSize:adjustingMag');
feature('DefaultCharacterSet', 'UTF8');

%% Get image list
images_dir = 'images';
listing = cat(1, dir(fullfile(images_dir, '*_input.jpg')), ...
    dir(fullfile(images_dir, '*.CR2')), ...
    dir(fullfile(images_dir, '*.jpg')), ...
    dir(fullfile(images_dir, '*.png')));

% Set up result dir
result_dir = fullfile(images_dir, 'results');
if ~exist(result_dir, 'dir'), mkdir(result_dir); end

% Colormap for transmission.
jetmap = jet(256);
verbose = false;

% Too large image will be resized
max_width = 2010;

%% Restoration
for i_img = 1:length(listing)
    [img_out, trans_out, A, estimated_water_type] = uw_restoration(...
        listing(i_img).name, listing(i_img).folder, edges_path, max_width, ...
        result_dir, verbose);

    [~, img_name, ~] = fileparts(listing(i_img).name);
    img_name = strrep(img_name, '_input', '');


    % Save the enhanced image and the transmission map.
    imwrite(im2uint8(img_out), fullfile(result_dir, [img_name, '_output_img.jpg']));
    imwrite(im2uint8(trans_out), jetmap, fullfile(result_dir, [img_name, '_output_map.jpg']));
end
