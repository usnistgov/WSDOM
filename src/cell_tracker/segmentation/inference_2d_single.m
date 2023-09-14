function img_seg = inference_2d_single(load_dir, filename, Trained_Net, save_dir, save_fname, normalize)

if ~exist("normalize", "var"), normalize = true; end
if ~isfolder(save_dir), mkdir(save_dir); end

% load the model
tile_size       = Trained_Net.Layers(1, 1).InputSize;
neighbor_size   = 32;

try
    img = imread(fullfile(load_dir, filename));
catch
    warning("Image not found");
    return
end

% fprintf("Inferencing %s phase image\n", filename);

if normalize, img = normalize_image(img, "zscore"); end
img = im2uint8(img);
[num_rows, num_cols] = size(img); 

img_pad = padarray(img, [neighbor_size, neighbor_size]);

ni = num_rows / tile_size(1);
nj = num_cols / tile_size(2);
% Check if we need to handle the last column and the last row, when the number of tiles with overlap does not equal the width or hight of the image
handle_row = 0;
handle_col = 0;
if floor(ni) ~= ni, handle_row = 1; end
if floor(nj) ~= nj, handle_col = 1; end
ni = floor(ni); % make sure number is int
nj = floor(nj); % make sure number is int


img_seg = zeros(num_rows, num_cols, 'uint8');
for i = 1:ni
    for j = 1:nj
        % Extract the tile with extra neighboring pixels
        i_min = (i-1)*tile_size(1)+1;
        i_max = i*tile_size(1);
        j_min = (j-1)*tile_size(2)+1;
        j_max = j*tile_size(2);
        I1 = img_pad(i_min:i_max+2*neighbor_size, j_min:j_max+2*neighbor_size);
        % Segment using UNet model
        S1 = semanticseg(I1,Trained_Net,'OutputType','uint8');
        % Save in correct place in the whole image
        img_seg(i_min:i_max, j_min:j_max) = S1(neighbor_size+1:neighbor_size+tile_size(1),neighbor_size+1:neighbor_size+tile_size(2));
    end
    % Handle last column
    if handle_col
        i_min = (i-1)*tile_size(1)+1;
        i_max = i*tile_size(1);
        j_min = num_cols-tile_size(2)+1;
        j_max = num_cols;
        I1 = img_pad(i_min:i_max+2*neighbor_size, j_min:j_max+2*neighbor_size);
        % Segment using UNet model
        S1 = semanticseg(I1,Trained_Net,'OutputType','uint8');
        % Save in correct place in the whole image
        img_seg(i_min:i_max, j_min:j_max) = S1(neighbor_size+1:neighbor_size+tile_size(1),neighbor_size+1:neighbor_size+tile_size(2));
    end
end

% Handle last row
if handle_row
    for j = 1:nj
        % Extract the tile with extra neighboring pixels
        i_min = num_rows-tile_size(1)+1;
        i_max = num_rows;
        j_min = (j-1)*tile_size(2)+1;
        j_max = j*tile_size(2);
        I1 = img_pad(i_min:i_max+2*neighbor_size, j_min:j_max+2*neighbor_size);
        % Segment using UNet model
        S1 = semanticseg(I1,Trained_Net,'OutputType','uint8');
        % Save in correct place in the whole image
        img_seg(i_min:i_max, j_min:j_max) = S1(neighbor_size+1:neighbor_size+tile_size(1),neighbor_size+1:neighbor_size+tile_size(2));
    end
    % Handle last column and last row, that lower corner tile!
    if handle_col
        i_min = num_rows-tile_size(1)+1;
        i_max = num_rows;
        j_min = num_cols-tile_size(2)+1;
        j_max = num_cols;
        I1 = img_pad(i_min:i_max+2*neighbor_size, j_min:j_max+2*neighbor_size);
        % Segment using UNet model
        S1 = semanticseg(I1,Trained_Net,'OutputType','uint8');
        % Save in correct place in the whole image
        img_seg(i_min:i_max, j_min:j_max) = S1(neighbor_size+1:neighbor_size+tile_size(1),neighbor_size+1:neighbor_size+tile_size(2));
    end
end
% Save image to disk
img_seg = uint8(img_seg-1); % put background to 0

% Save the image
outfilename = save_fname;
if isempty(save_fname), outfilename = filename; end
imwrite(img_seg, fullfile(save_dir, outfilename));
end

