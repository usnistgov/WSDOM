function inference_2d(img_dir, model_path, model_name, save_model_name, normalize)
    load(fullfile(model_path, model_name), 'Trained_Net');
    fprintf("Using model %s \n", model_name);
    load_dir   = fullfile(img_dir, 'phase');
    save_dir   = fullfile(img_dir, ['phase', '_inferenced'], save_model_name);
    mkdir(save_dir);
    
    filenames = dir(fullfile(load_dir, '*.tif'));
    parfor tt = 1:numel(filenames)
        load_fname = filenames(tt).name;
        save_fname = load_fname;
        if isfile(fullfile(save_dir, save_fname))
            fprintf("File exists\n");
            continue
        end
        inference_2D_single(load_dir, load_fname, Trained_Net, save_dir, save_fname, normalize);
    end
end