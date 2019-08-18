%% The demo for Cluster-based Co-saliency Detection in multiple images

clc;close all;clear;

addpath(genpath('A:\papernew/COS_code'));
addpath(genpath('A:\papernew'));


doFrameRemoving = true;
useSP = true;	% You can set useSP = false to use regular grid for speed consideration
useGuidedfilter = true; % You can set useGuidedfilter = true to smooth the image
doMAEEval = true;       % Evaluate MAE measure after saliency map calculation
doPRCEval = true;        % Evaluate PR Curves after saliency map calculation

%% image set 
para.img_set_name= 'Guo';
para.img_path=['A:\papernew\data\input\', para.img_set_name, '/'];
para.result_path = ['A:\papernew\img_output\', para.img_set_name, '/'];

if (~exist(para.result_path, 'dir')) 
    mkdir(para.result_path);
end

%% co-saliency parameters
para.files_list=dir([para.img_path '*.jpg']);
para.img_num=length(para.files_list);
% image resize scale
para.Scale=200; 
%clustering number on multi-image
para.Bin_num=min(max(2 * para.img_num,10),30);
%clustering number on single-image
para.Bin_num_single=6;

%% read images
data.image = cell(para.img_num,1);

for img_idx = 1:para.img_num
   
   imgtiffff = imread([para.img_path, para.files_list(img_idx).name]);
    
   data.image{img_idx} = imgtiffff(:,:,1:3);
end

%% cosaliency detection
 data=data; img_num=para.img_num; Scale=para.Scale; Bin_num= para.Bin_num;
 
 %% ------ Obtain the co-saliency for multiple images-------------
%----- obtaining the features -----
All_vector = [];
All_DisVector = [];
All_img = [];
for i=1:img_num
    img = data.image{i};
    [imvector, temp_img, DisVector]=GetImVector(img, Scale, Scale,1);
      imgR = imresize(img,[Scale Scale],'bilinear');
        
       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%EDGEBOX%%%%%%%%%%%%%%%%%%%%%%
        % Demo for Structured Edge Detector (please see readme.txt first).

    %% set opts for training (see edgesTrain.m)
    opts=edgesTrain();                % default options (good settings)
    opts.modelDir='models/';          % model will be in models/forest
    opts.modelFnm='modelBsds';        % model name
    opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
    opts.useParfor=0;                 % parallelize if sufficient memory

    %% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
    tic, model=edgesTrain(opts); toc; % will load model if already trained

    %% set detection parameters (can set after training)
    model.opts.multiscale=0;          % for top accuracy set multiscale=1
    model.opts.sharpen=2;             % for top speed set sharpen=0
    model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
    model.opts.nThreads=4;            % max number threads for evaluation
    model.opts.nms=0;                 % set to true to enable nms

    %% evaluate edge detector on BSDS500 (see edgesEval.m)
    if(0), edgesEval( model, 'show',1, 'name','' ); end

    %% detect edge and visualize results
    I = imgR;
    tic, E=edgesDetect(I,model);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Evalu= reshape(E,[Scale*Scale 1]);
    imvector(:,5)=Evalu;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%HOG%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    imgfv=zeros(Scale,Scale);
    [featureVector,hogVisualization] = extractHOGFeatures(imgR,'CellSize',[4 4]);
    [xfv,yfv]=size(featureVector);
    featureVectornew = reshape(featureVector,[sqrt(yfv) sqrt(yfv)]);
    for fvmax=1:sqrt(yfv)/6
        for fvmay =1:sqrt(yfv)/6
            maxfvtmp=6;
            fvtemp=4;
%              maxfvmatrix = max(max(featureVectornew((fvmax-1)*(maxfvtmp)+1:fvmax*(maxfvtmp),(fvmay-1)*(maxfvtmp)+1:fvmay*(maxfvtmp))));
%              minfvmatrix = min(min(featureVectornew((fvmax-1)*(maxfvtmp)+1:fvmax*(maxfvtmp),(fvmay-1)*(maxfvtmp)+1:fvmay*(maxfvtmp))));
%              sumfvmatrix = sum(sum(featureVectornew((fvmax-1)*(maxfvtmp)+1:fvmax*(maxfvtmp),(fvmay-1)*(maxfvtmp)+1:fvmay*(maxfvtmp))));
%              imgfv((fvmax-1)*(fvtemp)+1:fvmax*(fvtemp),(fvmay-1)*(fvtemp)+1:fvmay*(fvtemp))=sumfvmatrix/36;
                maxfvmatrix = max(max(featureVectornew((fvmax-1)*(maxfvtmp)+1:fvmax*(maxfvtmp),(fvmay-1)*(maxfvtmp)+1:fvmay*(maxfvtmp))));
                 imgfv((fvmax-1)*(fvtemp)+1:fvmax*(fvtemp),(fvmay-1)*(fvtemp)+1:fvmay*(fvtemp))=maxfvmatrix;
    
        end
    end
    imgfvnew=reshape(imgfv,[Scale*Scale 1]);
   imvector(:,8) = imgfvnew;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Basline%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing: Remove Image Frames
    srcImg = imgR;
    if doFrameRemoving
        [noFrameImg, frameRecord] = removeframe(srcImg, 'sobel');
        [h, w, chn] = size(noFrameImg);
    else
        noFrameImg = srcImg;
        [h, w, chn] = size(noFrameImg);
        frameRecord = [h, w, 1, h, 1, w];
    end
    
    if useGuidedfilter
        noFrameImg = imguidedfilter(noFrameImg);
    end
    
    %% create superpixel and graph
    sp_graph_prop = SuperpixelPropertyAndGraph(noFrameImg, useSP, 600, 250);    
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(sp_graph_prop.adjcMatrix, sp_graph_prop.colDistM);    
    
    %% Saliency Baseline
    centSigma = min(h, w) / 1000;
	% use the sigmoid function to enhance C_bnd, which achieves better performance but lower speed in large datasets.
	% set useSigmoid = true to achieve the same result as reported in ACCV 2014.
	useSigmoid = false;
    
    baseline = SaliencyBaseline(sp_graph_prop, clipVal, geoSigma, centSigma, useSigmoid);
     [stage2, stage1, bsalt, bsalb, bsall, bsalr] = ManifoldRanking(sp_graph_prop.adjcMatrix, sp_graph_prop.idxImg, sp_graph_prop.bdIds, sp_graph_prop.colDistM);

    outputimgB=SaveSaliencyMap(stage2, sp_graph_prop.pixelList, frameRecord, true);
    
    
    imgBasli=reshape(outputimgB,[Scale*Scale 1]);
   imvector(:,6) = imgBasli;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%MR Saliency%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    [stage2, stage1, bsalt, bsalb, bsall, bsalr] = ManifoldRanking(sp_graph_prop.adjcMatrix, sp_graph_prop.idxImg, sp_graph_prop.bdIds, sp_graph_prop.colDistM);

    outputimgMR=SaveSaliencyMap(stage2, sp_graph_prop.pixelList, frameRecord, true);
   
       imgMR=reshape(outputimgMR,[Scale*Scale 1]);
   imvector(:,7) = imgMR;
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%feature%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

    
    minve = min(imvector);
    maxve = max(imvector);
%     imvector(:,1)=(imvector(:,1)-minve(1,1))/(maxve(1,1)-minve(1,1));
%     imvector(:,2)=(imvector(:,2)-minve(1,2))/(maxve(1,2)-minve(1,2));
%     imvector(:,3)=(imvector(:,3)-minve(1,3))/(maxve(1,3)-minve(1,3));
%     imvector(:,4)=(imvector(:,4)-minve(1,4))/(maxve(1,4)-minve(1,4));
%     imvector(:,5)=imvector(:,5);
 %  imvector(:,6)=(imvector(:,6)-minve(1,6))/(maxve(1,6)-minve(1,6));
%  
%      imvector(:,1)=exp(-abs(imvector(:,1).^2/(2*(maxve(1,1)^2))));
%     imvector(:,2)=exp(-abs(imvector(:,2).^2/(2*(maxve(1,2)^2))));
%     imvector(:,3)=exp(-abs(imvector(:,3).^2/(2*(maxve(1,3)^2))));
%     imvector(:,4)=exp(-abs(imvector(:,4).^2/(2*(maxve(1,4)^2))));
%     imvector(:,5)=exp(-abs(imvector(:,5).^2/(2*(maxve(1,5)^2))));
%         imvector(:,6)=exp(-abs(imvector(:,6).^2/(2*(maxve(1,6)^2))));
%          imvector(:,7)=exp(-abs(imvector(:,7).^2/(2*(maxve(1,7)^2))));
%   %    imvector(:,8)=exp(-abs(imvector(:,8).^2/(maxve(1,8)^2)));
  
       imvector(:,1)=exp(-abs(imvector(:,1).^2/(2*(maxve(1,1)^2))));
    imvector(:,2)=exp(-abs(imvector(:,2).^2/(2*(maxve(1,2)^2))));
    imvector(:,3)=exp(-abs(imvector(:,3).^2/(2*(maxve(1,3)^2))));
    imvector(:,4)=exp(-abs(imvector(:,4).^2/(2*(maxve(1,4)^2))));
    imvector(:,5)=exp(-abs(imvector(:,5).^2/(2*(maxve(1,5)^2))));
        imvector(:,6)=exp(-abs(imvector(:,6).^2/(2*(maxve(1,6)^2))));
         imvector(:,7)=exp(-abs(imvector(:,7).^2/(2*(maxve(1,7)^2))));
         imvector(:,8)=exp(-abs(imvector(:,8).^2/(2*(maxve(1,8)^2))));

%      imvector(:,1)=exp(-imvector(:,1).^2/200);
%      imvector(:,2)=exp(-imvector(:,2).^2/200);
%      imvector(:,3)=exp(-imvector(:,3).^2/200);
%      imvector(:,4)=exp(-imvector(:,4).^2/200);
%      imvector(:,5)=exp(-imvector(:,5).^2/200);
%      imvector(:,7)=exp(-imvector(:,7).^2/200);
%       imvector(:,6)=exp(-imvector(:,6).^2/200);
%       imvector(:,8)=exp(-imvector(:,8).^2/200);
   
     [xvt,yvt]=find(isnan(imvector));
    imvector(xvt,yvt)=0;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    All_vector=[All_vector; imvector];
    All_DisVector=[All_DisVector; DisVector];
    All_img=[All_img temp_img];
end

% ---- clustering via Kmeans++ -------
    [ctrs,idx] = fcmPP(All_vector',Bin_num);
    
    tempidx = zeros(1,Scale*Scale*img_num);
    for nc = 1:Scale*Scale*img_num
        [maxvalue]=find(idx(:,nc)==max(idx(:,nc)));
        [xmax,ymax]=size(maxvalue);
        if xmax>1
           tempnum = round((xmax-1)*rand)+1;
           tempidx(1,nc)=maxvalue(tempnum);
        else
           tempidx(1,nc) = maxvalue(1);
        end

    end
    
    idx=tempidx'; ctrs=ctrs;

%----- clustering idx map ---------
Cluster_Map = reshape(idx, Scale, Scale*img_num);

%----- computing the Contrast cue -------
Sal_weight_co= Gauss_normal(GetSalWeight( ctrs,idx ));
%----- computing the Spatial cue -------
Dis_weight_co= Gauss_normal(GetPositionW( idx, All_DisVector, Scale, Bin_num ));
%----- computing the Corresponding cue -------
co_weight_co= Gauss_normal(GetCoWeight( idx, Scale, Scale ));
 
%----- combining the Co-Saliency cues -----
SaliencyWeight=(Sal_weight_co .* Dis_weight_co .* co_weight_co);

%----- generating the co-saliency map -----
Saliency_Map_co = Cluster2img( Cluster_Map, SaliencyWeight, Bin_num);
 
 result.cos_map=Saliency_Map_co;result.All_img=All_img;
 
 

%% single sliency detection
for i=1:img_num
        img = data.image{i};
        [imvector, ~, DisVector]=GetImVector(img, Scale, Scale,1);
        imgR = imresize(img,[Scale Scale],'bilinear');
        
       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%EDGEBOX%%%%%%%%%%%%%%%%%%%%%%
        % Demo for Structured Edge Detector (please see readme.txt first).

    %% set opts for training (see edgesTrain.m)
    opts=edgesTrain();                % default options (good settings)
    opts.modelDir='models/';          % model will be in models/forest
    opts.modelFnm='modelBsds';        % model name
    opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
    opts.useParfor=0;                 % parallelize if sufficient memory

    %% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
    tic, model=edgesTrain(opts); toc; % will load model if already trained

    %% set detection parameters (can set after training)
    model.opts.multiscale=0;          % for top accuracy set multiscale=1
    model.opts.sharpen=2;             % for top speed set sharpen=0
    model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
    model.opts.nThreads=4;            % max number threads for evaluation
    model.opts.nms=0;                 % set to true to enable nms

    %% evaluate edge detector on BSDS500 (see edgesEval.m)
    if(0), edgesEval( model, 'show',1, 'name','' ); end

    %% detect edge and visualize results
    I = imgR;
    tic, E=edgesDetect(I,model);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Evalu= reshape(E,[Scale*Scale 1]);
    imvector(:,5)=Evalu;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%HOG%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    imgfv=zeros(Scale,Scale);
    [featureVector,hogVisualization] = extractHOGFeatures(imgR,'CellSize',[4 4]);
    [xfv,yfv]=size(featureVector);
    featureVectornew = reshape(featureVector,[sqrt(yfv) sqrt(yfv)]);
    for fvmax=1:sqrt(yfv)/6
        for fvmay =1:sqrt(yfv)/6
            maxfvtmp=6;
            fvtemp=4;
%              maxfvmatrix = max(max(featureVectornew((fvmax-1)*(maxfvtmp)+1:fvmax*(maxfvtmp),(fvmay-1)*(maxfvtmp)+1:fvmay*(maxfvtmp))));
%              minfvmatrix = min(min(featureVectornew((fvmax-1)*(maxfvtmp)+1:fvmax*(maxfvtmp),(fvmay-1)*(maxfvtmp)+1:fvmay*(maxfvtmp))));
%              sumfvmatrix = sum(sum(featureVectornew((fvmax-1)*(maxfvtmp)+1:fvmax*(maxfvtmp),(fvmay-1)*(maxfvtmp)+1:fvmay*(maxfvtmp))));
%              imgfv((fvmax-1)*(fvtemp)+1:fvmax*(fvtemp),(fvmay-1)*(fvtemp)+1:fvmay*(fvtemp))=sumfvmatrix/36;
                maxfvmatrix = max(max(featureVectornew((fvmax-1)*(maxfvtmp)+1:fvmax*(maxfvtmp),(fvmay-1)*(maxfvtmp)+1:fvmay*(maxfvtmp))));
                 imgfv((fvmax-1)*(fvtemp)+1:fvmax*(fvtemp),(fvmay-1)*(fvtemp)+1:fvmay*(fvtemp))=maxfvmatrix;
    
        end
    end
    imgfvnew=reshape(imgfv,[Scale*Scale 1]);
   imvector(:,8) = imgfvnew;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Basline%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing: Remove Image Frames
    srcImg = imgR;
    if doFrameRemoving
        [noFrameImg, frameRecord] = removeframe(srcImg, 'sobel');
        [h, w, chn] = size(noFrameImg);
    else
        noFrameImg = srcImg;
        [h, w, chn] = size(noFrameImg);
        frameRecord = [h, w, 1, h, 1, w];
    end
    
    if useGuidedfilter
        noFrameImg = imguidedfilter(noFrameImg);
    end
    
    %% create superpixel and graph
    sp_graph_prop = SuperpixelPropertyAndGraph(noFrameImg, useSP, 600, 250);    
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(sp_graph_prop.adjcMatrix, sp_graph_prop.colDistM);    
    
    %% Saliency Baseline
    centSigma = min(h, w) / 1000;
	% use the sigmoid function to enhance C_bnd, which achieves better performance but lower speed in large datasets.
	% set useSigmoid = true to achieve the same result as reported in ACCV 2014.
	useSigmoid = false;
    
    baseline = SaliencyBaseline(sp_graph_prop, clipVal, geoSigma, centSigma, useSigmoid);
     [stage2, stage1, bsalt, bsalb, bsall, bsalr] = ManifoldRanking(sp_graph_prop.adjcMatrix, sp_graph_prop.idxImg, sp_graph_prop.bdIds, sp_graph_prop.colDistM);

    outputimgB=SaveSaliencyMap(stage2, sp_graph_prop.pixelList, frameRecord, true);
    
    
    imgBasli=reshape(outputimgB,[Scale*Scale 1]);
   imvector(:,6) = imgBasli;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%MR Saliency%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    [stage2, stage1, bsalt, bsalb, bsall, bsalr] = ManifoldRanking(sp_graph_prop.adjcMatrix, sp_graph_prop.idxImg, sp_graph_prop.bdIds, sp_graph_prop.colDistM);

    outputimgMR=SaveSaliencyMap(stage2, sp_graph_prop.pixelList, frameRecord, true);
   
       imgMR=reshape(outputimgMR,[Scale*Scale 1]);
   imvector(:,7) = imgMR;
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%feature%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

    
    minve = min(imvector);
    maxve = max(imvector);
%     imvector(:,1)=(imvector(:,1)-minve(1,1))/(maxve(1,1)-minve(1,1));
%     imvector(:,2)=(imvector(:,2)-minve(1,2))/(maxve(1,2)-minve(1,2));
%     imvector(:,3)=(imvector(:,3)-minve(1,3))/(maxve(1,3)-minve(1,3));
%     imvector(:,4)=(imvector(:,4)-minve(1,4))/(maxve(1,4)-minve(1,4));
%     imvector(:,5)=imvector(:,5);
 %  imvector(:,6)=(imvector(:,6)-minve(1,6))/(maxve(1,6)-minve(1,6));
%  
%      imvector(:,1)=exp(-abs(imvector(:,1).^2/(2*(maxve(1,1)^2))));
%     imvector(:,2)=exp(-abs(imvector(:,2).^2/(2*(maxve(1,2)^2))));
%     imvector(:,3)=exp(-abs(imvector(:,3).^2/(2*(maxve(1,3)^2))));
%     imvector(:,4)=exp(-abs(imvector(:,4).^2/(2*(maxve(1,4)^2))));
%     imvector(:,5)=exp(-abs(imvector(:,5).^2/(2*(maxve(1,5)^2))));
%         imvector(:,6)=exp(-abs(imvector(:,6).^2/(2*(maxve(1,6)^2))));
%          imvector(:,7)=exp(-abs(imvector(:,7).^2/(2*(maxve(1,7)^2))));
%   %    imvector(:,8)=exp(-abs(imvector(:,8).^2/(maxve(1,8)^2)));
  
       imvector(:,1)=exp(-abs(imvector(:,1).^2/(2*(maxve(1,1)^2))));
    imvector(:,2)=exp(-abs(imvector(:,2).^2/(2*(maxve(1,2)^2))));
    imvector(:,3)=exp(-abs(imvector(:,3).^2/(2*(maxve(1,3)^2))));
    imvector(:,4)=exp(-abs(imvector(:,4).^2/(2*(maxve(1,4)^2))));
    imvector(:,5)=exp(-abs(imvector(:,5).^2/(2*(maxve(1,5)^2))));
        imvector(:,6)=exp(-abs(imvector(:,6).^2/(2*(maxve(1,6)^2))));
         imvector(:,7)=exp(-abs(imvector(:,7).^2/(2*(maxve(1,7)^2))));
         imvector(:,8)=exp(-abs(imvector(:,8).^2/(2*(maxve(1,8)^2))));

%      imvector(:,1)=exp(-imvector(:,1).^2/200);
%      imvector(:,2)=exp(-imvector(:,2).^2/200);
%      imvector(:,3)=exp(-imvector(:,3).^2/200);
%      imvector(:,4)=exp(-imvector(:,4).^2/200);
%      imvector(:,5)=exp(-imvector(:,5).^2/200);
%      imvector(:,7)=exp(-imvector(:,7).^2/200);
%       imvector(:,6)=exp(-imvector(:,6).^2/200);
%       imvector(:,8)=exp(-imvector(:,8).^2/200);
   
     [xvt,yvt]=find(isnan(imvector));
    imvector(xvt,yvt)=0;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%fuzzy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %[idx,ctrs] = kmeansPP(imvector',Bin_num_single);
    [ctrs,idx] = fcmPP(imvector',Bin_num);
    
     maxUidx = max(idx);
    tempidx = zeros(1,Scale*Scale);
    for nc = 1:Scale*Scale
        [maxvalue]=find(idx(:,nc)==max(idx(:,nc)));
        [xmax,ymax]=size(maxvalue);
        if xmax>1
           tempnum = round((xmax-1)*rand)+1;
           tempidx(1,nc)=maxvalue(tempnum);
        else
           tempidx(1,nc) = maxvalue(1);
        end

    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    idx=tempidx';
     maxUidx_matrix = reshape(maxUidx,Scale, Scale );
    Cluster_Map = reshape(idx, Scale, Scale);
    Sal_weight=GetSalWeight( ctrs,idx );
    Dis_weight  = GetPositionW( idx, DisVector, Scale, Bin_num );
      Sal_weight= Gauss_normal(Sal_weight);
        Dis_weight= Gauss_normal(Dis_weight);
    SaliencyWeight_all=(Sal_weight .* Dis_weight);
      SaliencyWeight_all=Gauss_normal(SaliencyWeight_all);
    Saliency_sig_final = Cluster2img( Cluster_Map, SaliencyWeight_all, Bin_num);
    
       Saliency_Map_single(:,1+(i-1)*Scale:Scale+(i-1)*Scale)=exp(Saliency_sig_final.^2/(2*20^2));
  % Saliency_Map_single(:,1+(i-1)*Scale:Scale+(i-1)*Scale)=Saliency_sig_final;
end
result.single_map = Gauss_normal( Saliency_Map_single);

%% combine saliency map
result.final_map = result.single_map .* result.cos_map*2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath( genpath( '.' ) );
foldername = fileparts( mfilename( 'fullpath' ) );

options.valScale = 60;
options.alpha = 0.05;
options.color_size = 5;
%% Print status messages on screen
options.vocal = true;
options.regnum =500;
options.m = 20;
options.gradLambda = 1;
videoFiles = dir(fullfile(foldername, 'data', 'input'));
videoNUM = length(videoFiles)-2;


for videonum = 1:videoNUM
    videofolder =  videoFiles(videonum+2).name;
    options.infolder = fullfile( foldername, 'data', 'input',videofolder );
    % The folder where all the outputs will be stored.
    options.outfolder = fullfile( foldername, 'data', 'output2', videofolder );
    if( ~exist( options.outfolder, 'dir' ) )
        mkdir( options.outfolder ),
    end;
    if( ~exist( fullfile( options.outfolder, 'energy'), 'dir' ) )
        mkdir(fullfile( options.outfolder, 'energy'));
    end
    if( ~exist( fullfile( options.outfolder, 'saliency'), 'dir' ) )
        mkdir(fullfile( options.outfolder, 'saliency'));
    end
    % Cache all frames in memory
    [data.frames,data.names,height,width,nframe ]= readAllFrames( options );
    
     % Load optical flow (or compute if file is not found)
    data.flow = loadFlow( options );
    if( isempty( data.flow ) )
        data.flow = computeOpticalFlow( options, data.frames );
    end
    
    % Load superpixels (or compute if not found)
    data.superpixels = loadSuperpixels( options );
    if( isempty( data.superpixels ) )
        data.superpixels = computeSuperpixels(  options, data.frames );
    end
    [ superpixels, nodeFrameId, bounds, labels ] = makeSuperpixelIndexUnique( data.superpixels );
    [ colours, centres, t ] = getSuperpixelStats( data.frames(1:nframe-1), superpixels, double(labels) );%
    
    valLAB = [];
    for index = 1:nframe-1
        valLAB = [valLAB;data.superpixels{index}.Sup1, data.superpixels{index}.Sup2, data.superpixels{index}.Sup3];     
    end
    RegionSal = [];
    frameEnergy = cell( nframe-1, 1 );

    foregroundArea = 0;
    for index = 1:nframe-1        
        frame = data.frames{index};        
        frameName = data.names{index};    
        nLabel = max(data.superpixels{index}.Label(:));
        Label = data.superpixels{index}.Label;
        framex = reshape(frame,height*width,1,3);
        Label = reshape(Label,height*width,1);       
        frameVal = colours(bounds(index):bounds(index+1)-1,:);
        framex = uint8(reshape(superpixel2pixel(double(data.superpixels{index}.Label),double(frameVal)),height ,width,3));       
        framex=imfilter(framex,fspecial('average',3),'same','replicate');
        G = edge_detect(framex);%static boundary

        gradient = getFlowGradient( data.flow{index} );
        magnitude = getMagnitude( gradient );

        if index>1
            mask = imdilate((frameEnergy{index-1}>0.3),strel('diamond',30))+0.3;
            mask(mask(:)>1)=1;
            magnitude = magnitude.*mask;
            G = G.*mask;
        end

        gradBoundary = 1 - exp( -options.gradLambda * magnitude );        
        
        if (max(magnitude(:))<10)
            gradBoundary = gradBoundary + 0.01;
        end
        
        G = G.*( gradBoundary );%spatio-temporal gradient
        
        %% saliency via gradient flow
        [V_Energy1 H_Energy1 V_Energy2 H_Energy2] = energy_map(double(framex),double(G));
        
        if index ==1
            Energy = min(min(min(H_Energy1,V_Energy1),H_Energy2),V_Energy2);       
        else
            mask = int32(imdilate((Energy>0.2),strel('diamond',20)));
            mask = ~mask;
            Energymap = (Energy<0.05).*mask; 
            Energymap = ~Energymap;
            Energy = Energy*0.3+(Energymap.*min(min(min(H_Energy1,V_Energy1),H_Energy2),V_Energy2))*0.7;%considering saliency of prior frame
        end
        
        Energy = Energy/max(Energy(:));         
        L{1} = uint32(data.superpixels{index}.Label);
        S{1} = repmat(Energy,[1 3]);
        [ R, ~, ~ ] = getSuperpixelStats(S(1:1),L, double(nLabel) );
        R = double(R(:,1));
        [sR,indexR] = sort(R);
        t = sum(sR(end-9:end))/10;
        R = (R-min(R))/(t-min(R));
        R(R>1)=1;
        RegionSal = [RegionSal;R];
        Energy = reshape(superpixel2pixel(double(data.superpixels{index}.Label),double(R)),height ,width); 
        imwrite(Energy, [options.outfolder '\energy\' 'initial_' frameName  '.bmp']);
        frameEnergy{index} = Energy;
        foregroundArea = foregroundArea + sum(sum(frameEnergy{index}>0.6));   
    end
    %% large salient object
    foregroundArea = foregroundArea/(nframe-1);
    if foregroundArea > height*width*0.02
        for index = 1:nframe-1
            Energymap = ones(height,width);
            Label = data.superpixels{index}.Label;
            if index ==1
                mask1 = int32(imdilate((frameEnergy{index}>0.4),strel('diamond',20)));
                mask1 = ~mask1;
                mask2 = int32(imdilate((frameEnergy{index+1}>0.4),strel('diamond',20)));
                mask2 = ~mask2;
                mask = mask1.*mask2;
                mask(1:end,1) = 1;
                mask(1:end,end) = 1;
                mask(1,1:end) = 1;
                mask(end,1:end) = 1;
                Energymap = Energymap.*mask;     
            else
                mask1 = int32(imdilate((frameEnergy{index}>0.4),strel('diamond',20)));
                mask1 = ~mask1;
                mask2 = int32(imdilate((frameEnergy{index-1}>0.4),strel('diamond',20)));
                mask2 = ~mask2;
                mask = mask1.*mask2;
                mask(1:end,1) = 1;
                mask(1:end,end) = 1;
                mask(1,1:end) = 1;
                mask(end,1:end) = 1;
                Energymap = Energymap.*mask;
            end
            labelnum = max(Label(:));
             [ ConSPix, ConSPix1, ConSPDouble ] = find_connect_superpixel_DoubleIn_Opposite( Label, labelnum, height ,width );
             
       %% background contrast%%%%%%%%%%%%%%%%
        EdgSup = int32(Energymap).*Label;
        EdgSup = unique(EdgSup(:));
        EdgSup(EdgSup==0) = [];
        foreground = ones(labelnum,1);
        foreground( EdgSup) = 0;
        background = zeros(labelnum,1);
        background( EdgSup) = 1;
     
            [edges_x edges_y] = find(triu(ConSPix1)>0);
            ConS = [edges_x edges_y];
            t = edges_x-edges_y;
            ConS(t==0,:) = [];
            DcolNor=sqrt(sum((valLAB(ConS(:,1)+bounds(index)-1,:)-valLAB(ConS(:,2)+bounds(index)-1,:)).^2,2));
            for i =1:size(ConS,1)
                if background(ConS(i,1))==1&&background(ConS(i,2))==1
                    DcolNor(i)=0.0001;
                end
            end
        WconFirst=sparse([ConS(:,1);ConS(:,2)],[ConS(:,2);ConS(:,1)], ...
             [ DcolNor; DcolNor],double(labelnum),double(labelnum))+ sparse(1:double(labelnum),1:double(labelnum),ones(labelnum,1));
        geoDis = graphallshortestpaths(WconFirst);
        geoDis(:,logical(foreground)) = [];
        geoDis(logical(background),:) = [];
        
        [edges_x edges_y] = find(triu(ones(labelnum,labelnum))>0);
        ConSPDouble = [edges_x edges_y];
        colorDis=sqrt(sum((valLAB(ConSPDouble(:,1)+bounds(index)-1,:)-valLAB(ConSPDouble(:,2)+bounds(index)-1,:)).^2,2));      
        colorDis=sparse([ConSPDouble(:,1);ConSPDouble(:,2)],[ConSPDouble(:,2);ConSPDouble(:,1)], ...
             [ colorDis; colorDis],double(labelnum),double(labelnum));
        colorDis = full(colorDis);
        colorDis(:,logical(foreground)) = [];
        colorDis(logical(background),:) = [];
        posDis=double(sqrt(sum((centres(ConSPDouble(:,1)+bounds(index)-1,:)-centres(ConSPDouble(:,2)+bounds(index)-1,:)).^2,2)));      
        posDis=sparse([ConSPDouble(:,1);ConSPDouble(:,2)],[ConSPDouble(:,2);ConSPDouble(:,1)], ...
             [ posDis; posDis],double(labelnum),double(labelnum));
        posDis = full(posDis);
        posDis(:,logical(foreground)) = [];
        posDis(logical(background),:) = [];
        u = 2*min(posDis')+0.001;
        u =  repmat(u',1,size(posDis,2));
        posDis = exp(-posDis./u);
        geoSal = normalize(sum(geoDis,2));%
        contrastSal = normalize(sum(colorDis.*posDis,2)./sum(posDis,2));%+
        foreSal = geoSal.*min(contrastSal,0.5);
        
        if  size(foreSal,1) > 40
            [sR,indexR] = sort(foreSal);
            t = sum(sR(end-4:end))/5;
        else
            [sR,indexR] = sort(foreSal);
            t = double(sum(sR(end-(int32(size(foreSal,1)*0.1))+1:end))/double((int32(size(foreSal,1)*0.1))));     
        end
        foreSal = (foreSal-min(foreSal))/(t-min(foreSal));
        foreSal(foreSal>1)=1;
        foreSal = normalize(foreSal);
        Sal = zeros(labelnum,1);
        Sal(logical(foreground))=foreSal;

        Sal = 0.3*RegionSal(bounds(index):bounds(index+1)-1) + 0.7*Sal;
        Energy = reshape(superpixel2pixel(double(Label),double(Sal)),height ,width); 
        imwrite(Energy, [options.outfolder '\energy\' 'ref_' data.names{index} '.bmp']);
        RegionSal(bounds(index):bounds(index+1)-1)=Sal;
        frameEnergy{index} = Energy;
        end
        clear EdgWcon I N WconFirst E iD P
    end
    
    
    %% Spatiotemporal consistency%%%%%%%%%%%%
        ConSPix = []; Conedge = [];         
        for index = 1:nframe-1
            Label = data.superpixels{index}.Label;
            [conSPix conedge]= find_connect_superpixel( Label, max(Label(:)), height ,width );      
            Conedge = [Conedge;conedge + bounds(index)-1];
        end
        intralength = size(Conedge,1);
        for index = 1:nframe-2
            [x y] = meshgrid(1:bounds(index+1)-bounds(index),1:bounds(index+2)-bounds(index+1));
            conedge = [x(:)+bounds(index)-1,y(:)+bounds(index+1)-1];
            connect = sum((centres(conedge(:,1),:) - centres(conedge(:,2),:)).^2,2 );
            Conedge = [Conedge;conedge(find(connect<800),:)];
        end

    valDistances=sqrt(sum((valLAB(Conedge(:,1),:)-valLAB(Conedge(:,2),:)).^2,2));
    valDistances(intralength+1:end)=valDistances(intralength+1:end)/5;
    valDistances=normalize(valDistances);
    weights=exp(-options.valScale*valDistances)+ 1e-5;
    weights=sparse([Conedge(:,1);Conedge(:,2)],[Conedge(:,2);Conedge(:,1)], ...
    [weights;weights],labels,labels);
    E = sparse(1:labels,1:labels,ones(labels,1)); iD = sparse(1:labels,1:labels,1./sum(weights));
    P = iD*weights;
    
    RegionSal = (E-P+10*options.alpha*E)\RegionSal;
    
   %% generating final saliency 
    for index = 1:nframe-1
        frameName = data.names{index};
        Label = data.superpixels{index}.Label;
        R = RegionSal(bounds(index):bounds(index+1)-1);
        [sR,indexR] = sort(R);
        t = sum(sR(end-9:end))/10;
        R = (R-min(R))/(t-min(R));
        %R(R>1)=1;
        
        Energy = reshape(superpixel2pixel(double(Label),double(R)),height,width); 
        
        Energytest = imresize(Energy,[para.Scale,para.Scale]);
        
        Energytest_Map_single(:,1+(index-1)*Scale:Scale+(index-1)*Scale)=Energytest;
        imwrite(Energy, [options.outfolder '\saliency\' frameName '.bmp']);
    end
        Energytest_Map_single(:,1+(nframe-1)*Scale:Scale+(nframe-1)*Scale)=Energytest;
end


result.final_map=result.final_map.*Energytest_Map_single*2;




%% save the results
figure(1),subplot(3,1,1), imshow(result.All_img),title('Input images');
subplot(3,1,2), imshow(result.single_map),colormap(gray),title('Single Saliency');
subplot(3,1,3), imshow(result.final_map),colormap(gray),title('Co-Saliency');
Save_result( data, para, result);






