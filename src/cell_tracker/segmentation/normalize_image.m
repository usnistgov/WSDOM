

function I = normalize_image(I,normalization_choice)

if nargin == 1
    normalization_choice = "zscore";
end

% zscore normalize
if string(normalization_choice) == "zscore"
    I = double(I);
    I = (I-mean2(I))/std2(I);
    
    rangeMin = max(min(I(:)),-5);
    rangeMax = min(max(I(:)), 5);
    
    I(I > rangeMax) = rangeMax;
    I(I < rangeMin) = rangeMin;
    
    % Rescale the data to the range [0, 1].
    I = (I - rangeMin) / (rangeMax - rangeMin);
end

% zerocenter normalization
if string(normalization_choice) == "zerocenter"
    I = double(I);
    I = (I-mean2(I));
    % Rescale the data to the range [0, max].
    I = I + min(I(:));
end

% MinMax normalization
if string(normalization_choice) == "minmax"
    c = string(class(I));
    m = 255;
    if c == "uint16", m = 65532; end
    if c == "single", m = 30000; end % specify value manually
    I = double(I);
    I = I/m;
end




