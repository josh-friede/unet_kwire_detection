%% Input Wire Template(s) & Source Image Data.
% Get wire template(s).
wirePathName	= uigetdir('*.*','Select Wire Template(s).');
addpath(wirePathName);

% Load source image data directory.
imgPathName	= uigetdir(wirePathName, 'Select Source Image Folder');
d	= dir(imgPathName);
imgFolderNames	= fullfile(imgPathName, {d(3:end).name}');

% Get data files - we want 50-50 distribution between non-dhs images.
rndnumseed	= randi(99,1);
folderContents  = cell(length(imgFolderNames), 1);
dataFileNames   = [];
for idx = 1:length(imgFolderNames)
    filesInFolder   = dir(imgFolderNames{idx});
    
    % 25-25-50 % distribution chest-hip-random.
    fileNames   = {filesInFolder(3:end).name}';
    if idx == 1 %hardcoded
        ifiles = randi(length(fileNames), 385, 1);
        fileNames = fileNames(ifiles);
        
    elseif idx == 3
        ifiles = randi(length(fileNames), 385*2, 1);
        fileNames = fileNames(ifiles);
    end
    folderNames	= repmat(imgFolderNames{idx}, length(fileNames), 1);
    
    fileSeperators  = repmat(filesep, length(fileNames), 1);
    dataFileNames   = [dataFileNames; strcat(folderNames,fileSeperators, fileNames)];
end
numImages	= length(dataFileNames);

% Select saving destination.
saveDir	= uigetdir(imgPathName, 'Select a save directory.');

% Create training, testing data folders.
xTrainPath  = fullfile(saveDir,'xTrain');
yTrainPath  = fullfile(saveDir,'yTrain');
if ~exist(xTrainPath,'dir')
    [~,~,~] = mkdir(saveDir,'xTrain');
end
if ~exist(yTrainPath,'dir')
    [~,~,~] = mkdir(saveDir,'yTrain');
end
xTrainPath  = fullfile(xTrainPath, 'xTrain_');
yTrainPath  = fullfile(yTrainPath, 'yTrain_');

%% User Inputs.
%**************************************************************************
%**************************Change these if needed**************************
% Initialize augmentation and saving parameters
numAugs	= 5;
imgSize	= [256 256];                                % Desired image size.
bitdepth	= 8;                                    % Image file resolution.
blackThresh	= [10 128];                            	% For coloring wire.
angularBend	= 16;                                   % For bending wire.
tFactor	= 0.5;
augmenter	= imageDataAugmenter(...
    'FillValue',0,...
    'RandXReflection', true, 'RandYReflection', true,...
    'RandRotation', [0 360],...
    'RandXTranslation', [-imgSize(2)*tFactor imgSize(2)*tFactor],...
    'RandYTranslation', [-imgSize(1)*tFactor imgSize(1)*tFactor]);%,...
%     'RandXScale', [0.80 1.20], 'RandYScale', [0.80 1.20],...% Don't do
%     this if you have multiple templates of varying size.
%     'RandXShear',[-5 5], 'RandYShear',[-5 5]);
%**************************************************************************

%% Create Bent Wire Template(s).
% Get wire info.
templateDir	= dir(wirePathName);
templateNames   = {templateDir.name}';
templateNames   = templateNames(3:end);
nTemplates	= length(templateNames);

% Create transformation function for bending wire.
t = @(x) x(:,1).*-pi/angularBend;
f = @(x) [x(:,1).*cos(t(x))+x(:,2).*sin(t(x)),-x(:,1).*sin(t(x))+x(:,2).*cos(t(x))];
g = @(x, unused) f(x);
tform = maketform('custom', 2, 2, [], g, []);

% Perform on wire template(s):
wireTemplates   = cell(nTemplates,1);
wireTemplatesBW = wireTemplates;
wireTemplates_bent   = wireTemplates;
wireTemplates_bentBW	= wireTemplates;
for idx = 1:nTemplates
    img = imread(templateNames{idx});
    
    % Ensure all are same size as desired training-image size.
    if all(size(img) == imgSize)
        wireTemplates{idx}	= img;
    else
        wireTemplates{idx}  = imresize(img, imgSize);
    end
     wireTemplatesBW{idx}	= imbinarize(wireTemplates{idx});
    % figure; imshow(wireTemplate{idx}); pause; close
     
    % Create a bent-wire template.
    wireTemplates_bent{idx} = imtransform(wireTemplates{idx}, tform,...
         'UData', [-1 1], 'VData', [-1 1],...
         'XData', [-1 1], 'YData', [-1 1]);
     wireTemplates_bentBW{idx}  = imbinarize(wireTemplates_bent{idx});
end

%% Apply Augmentations.
% Create random lists for various selections in each training image.
totalNumImages	= numImages*((2*numAugs)+1);
blackThreshRatio    = blackThresh/255;
rng('shuffle');
ind_wireTemplates   = repmat(1:nTemplates, 1,...        % Randomly-selected templates.
    ceil(totalNumImages/nTemplates));
ind_wireTemplates   = ind_wireTemplates(randperm(totalNumImages));
ind_wireTemplates_bent   = ind_wireTemplates(randperm(totalNumImages));
rng('shuffle');
ind_negatives	= randi(numImages, numImages/10, 1);    % 10% of images are negatives.
rng('shuffle');
ind_plotting	= randi(numImages, 10, 1);              % Plot 10 examples.

% Initialize data analysis variables.
centroidPerWire	= cell(totalNumImages*2,1);
dicePerWire	= zeros(totalNumImages*2,1);

% Iterate through each source image.
wb   = waitbar(0,'Starting');
iTrain 	= 1;
iWireAug	= 1;
tic;
for idx = 1:numImages
    % Read image files.
    img	= imresize( imread( strtrim(dataFileNames{idx})), imgSize);
    try
        img = rgb2gray(img);
    catch
    end
    negative	= im2uint8(img);
    iblack  = negative == 0;
    
    % Create negatives - leave yTrain alone as false.
    if ismember(idx,ind_negatives)
        imwrite(negative, strcat(xTrainPath, num2str(iTrain), '.jpg'),...
            'jpg', 'bitdepth', bitdepth);
        imwrite(false(imgSize(1)), strcat(yTrainPath, num2str(iTrain), '.jpg'),...
            'jpg', 'bitdepth', bitdepth);
        iTrain = iTrain + 1;
    end
    
    % Apply rand. img. data augs. to wire; try to not occur too oft -> larger IMAX.
    ind_addSecondWire	= unique(randi([0 numAugs+1], randi(numAugs, 1), 1));
    for jdx	= 1:numAugs
        % Select a wire template (regular and bent).
        wireTemplate    = wireTemplates{ind_wireTemplates(iTrain)};
        wireTemplateBW	= wireTemplatesBW{ind_wireTemplates(iTrain)};
        wireTemplate_bent	= wireTemplates_bent{ind_wireTemplates(iTrain)};
        wireTemplate_bentBW	= wireTemplates_bentBW{ind_wireTemplates(iTrain)};
        
        % Augment wire.
        augWire	= augment(augmenter, wireTemplate);
        augWire_bent	= augment(augmenter, wireTemplate_bent);
        
        % Binarize wire.
        augWireBW   = imbinarize(augWire);
        augWire_bentBW	= imbinarize(augWire_bent);
        
        % Compute dice coefficients and centroids.
        dicePerWire(iWireAug)	= dice(augWireBW, wireTemplateBW);
        dicePerWire(iWireAug+1)	= dice(augWire_bentBW, wireTemplate_bentBW);
        centroidPerWire{iWireAug}	= struct2cell(regionprops(augWireBW, 'centroid'));
        centroidPerWire{iWireAug+1}	= struct2cell(regionprops(augWire_bentBW, 'centroid'));
        iWireAug	= iWireAug + 2;
        
        % Add second wire.
        if ismember(jdx,ind_addSecondWire)
            % Add another wire to template, augment that too.
            augWire2	= augment(augmenter, wireTemplate);
            augWireBW2  = imbinarize(augWire2);
            augWire	= augWire + augWire2;
            augWireBW   = imbinarize(augWire);
            augWire_bent2	= augment(augmenter, wireTemplate_bent);
            augWire_bentBW2	= imbinarize(augWire_bent2);
            augWire_bent	= augWire_bent + augWire_bent2;
            augWire_bentBW	= imbinarize(augWire_bent);
                        
            % Compute dice coefficients and centroids.
            dicePerWire(iWireAug)	= dice(augWireBW2, wireTemplateBW);
            dicePerWire(iWireAug+1)	= dice(augWire_bentBW2, wireTemplate_bentBW);
            centroidPerWire{iWireAug}	= struct2cell(regionprops(augWireBW2, 'centroid'));
            centroidPerWire{iWireAug+1}	= struct2cell(regionprops(augWire_bentBW2, 'centroid'));
            iWireAug	= iWireAug + 2;
        end
        
        % Get ratio of "blackness" for each pixel.
        augWireRatio    = double(augWire)/255;
        augWireRatio(~augWireBW)	= 0;
        augWire_bentRatio = double(augWire_bent)/255;
        augWire_bentRatio(~augWire_bentBW)	= 0;
        
        % Change pixel values of wire so that it's not completely black.
        wireColor   = ones(imgSize).*randi(blackThresh, 1);
        augWireColor    = uint8(wireColor.*augWireRatio);
        augWire_bentColor    = uint8(wireColor.*augWire_bentRatio);
        
        % Overlay wire.
        overlaidWire   = imsubtract(negative, augWireColor);
        overlaidWire_bent	= imsubtract(negative, augWire_bentColor);
        augWireBW(iblack)	= 0;
        augWire_bentBW(iblack)	= 0;
        
        % View images every once in a while to make sure they match template.
        if ismember(idx,ind_plotting) && jdx == 1
%             f=figure('units','normalized','outerposition',[0 0 1 1]);
%             subplot(2,3,1);	imshow(overlaidWire);
%             subplot(2,3,2); imshow(overlaidWire_bent);
%             subplot(2,3,4); imshow(augWireBW);
%             subplot(2,3,5); imshow(augWire_bentBW);
%             subplot(2,3,3); imshow(poo);pause;close;
        end
        
        % Save images.
        imwrite(overlaidWire, strcat(xTrainPath, num2str(iTrain), '.jpg'),...
            'jpg','bitdepth', bitdepth);
        imwrite(augWireBW, strcat(yTrainPath, num2str(iTrain), '.jpg'),...
            'jpg', 'bitdepth', bitdepth);
        iTrain	= iTrain + 1;
        imwrite(overlaidWire_bent, strcat(xTrainPath, num2str(iTrain), '.jpg'),...
            'jpg','bitdepth', bitdepth);
        imwrite(augWire_bentBW, strcat(yTrainPath, num2str(iTrain), '.jpg'),...
            'jpg', 'bitdepth', bitdepth);
        iTrain	= iTrain + 1;
    end
    
    % Waitbar.
    totalNumImages = totalNumImages + length(ind_addSecondWire);
    tval = iTrain/totalNumImages;
    waitbar(tval, wb, ['Progress: ', num2str(tval*100),'%']);
end
delete(wb);  toc
disp(['total number of images: ', num2str(totalNumImages)])

%% Show Aata Analysis Results.
% Plot showing how dissimilar the templates are from the augmentations.
figure('color','w');
subplot(6, 6, [7:9, 13:15, 19:21, 25:27, 31:33])
histogram(dicePerWire, 'binedges', 0:.01:.1);
title('Augmentation''s Similarity to Template');
ylabel('# of Images');
xlabel('Dice-Coefficient');
grid on;

% Plot showing dispersion of wire centroids after augmentation.
copy	= centroidPerWire;% centroidPerWire = copy;
centroidPerWire = centroidPerWire(~cellfun(@isempty, centroidPerWire));
centroidPerWire = centroidPerWire(cellfun(@length, centroidPerWire) == 1);
x	= zeros(length(centroidPerWire), 1);
y	= zeros(length(centroidPerWire), 1);
for idx = 1:length(centroidPerWire)
    x(idx)	= centroidPerWire{idx}{1}(1);
    y(idx)	= centroidPerWire{idx}{1}(2);
end
subplot(6, 6, [10:12, 16:18, 22:24, 28:30, 34:36])
plot(x, y, 'b.');
title('Wire Augmentations'' Centroid Dispersion');
ylabel('Y');
xlabel('X');
xlim([min(x), max(x)]);
ylim([min(y), max(y)]);
axis square
set(findall(gcf,'-property','FontSize'),'FontSize',20);

%% Save Wire Template and Bent Wire Template Images.
bentTemplatesSaveDir	= uigetdir([],'Select a save directory to save bent templates to.');
for idx = 1:10
    bentWireTemplate	= augment(augmenter, wireTemplates_bent{idx});
    imwrite(bentWireTemplate, strcat(fullfile(bentTemplatesSaveDir,...
        'bent_'), num2str(idx), '.jpg'),...
        'jpg', 'bitdepth', bitdepth);
end
