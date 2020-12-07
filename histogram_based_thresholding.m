% --------------- Example How to Load and Work with our thermal data ------
addpath 'util';
clear all; clc; close all; % clean up!

%% Setup
% linenumber = 11;
% site = 'F10'; 
% image_nr = 6;
% thermalParams = load( './data/camParams_thermal.mat' ); %load intrinsics
% thermalpath = fullfile( datapath, 'Images', num2str(linenumber) ); % path to thermal images
% thermalds = datastore( thermalpath );

% Folder with a few sampled images
datapath = fullfile( './sample_data/'); 
lines = datastore(datapath,'IncludeSubfolders',true);

%% Area under Histogramm with Threshold value in %
threshold_value = 0.0015; 


for i=1:length(lines.Files)
    I = imread(lines.Files{i});
    max_I = double(max(I(:)));
    min_I = double(min(I(:)));
    I_norm = (double(I)-min_I)./(max_I-min_I);

    I_surpress = I_norm;
    I_surpress(I_norm < mean(I_norm(:)) + 2.5*std(I_norm(:))) = 0;
    fig_hist=figure; set(fig_hist,'visible','off');
    hist = histogram(I_norm,'Normalization', 'pdf');
    i = length(hist.Values);
    area = 0;
    while true

        if area < threshold_value
            area = hist.Values(i)*hist.BinWidth + area;
        else 
            break;
        end
        i = i-1;
    end

    SP = i/length(hist.Values);
    close(fig_hist)
    fig=figure; 
    hax=axes; 
    hold on
    imhist(I_norm);
    line([SP SP],get(hax,'YLim'),'Color','red','LineWidth',1.5)  
    title('Histogramm of normalized Image')
    hold off
    
    figure;
    imshowpair(I_norm,I_surpress,'montage')
end


%% Std and Mean Based Thresholding
for i=1:length(lines.Files)
    I = imread(lines.Files{i});
    max_I = double(max(I(:)));
    min_I = double(min(I(:)));
    I_norm = (double(I)-min_I)./(max_I-min_I);
  
    I_surpress = I_norm;
    I_surpress(I_norm < mean(I_norm(:)) + 2.5*std(I_norm(:))) = 0;
    fig=figure; 
    hax=axes; 
    hold on
    imhist(I_norm);
    hist = histogram(I_norm, 'Normalization', 'pdf');
    area = sum(hist.Values)*hist.BinWidth;
    SP=mean(I_norm(:)) + 3*std(I_norm(:)); 
    line([SP SP],get(hax,'YLim'),'Color','red','LineWidth',1.5)  
    title('Histogramm of normalized Image')
    hold off
    figure;
    imshowpair(I_norm,I_surpress,'montage')
    
    

%     % Calculate Adaptive threshold values with high sensitivity 
%     % means more pixels as foreground at the risk of including some background pixels
%     T = adaptthresh(I_surpress,1,'ForegroundPolarity','bright','Statistic','mean');
%     % Binarize Image
%     BW = imbinarize(I_surpress,T);
% 
%     % Extract objects from binary image using properties
%     % create image that contains only those regions in the original image that
%     % do not have holes (Euler = 1)
%     BW2 = bwpropfilt(BW,'EulerNumber',[1 1]);
% 
%     % create image that contains only those regions in the original image that
%     % have an area between 10 and 400 pixels
%     BW3 = bwareafilt(BW2,[10 400]);




%     figure
%     montage({I_norm,I_surpress,BW,BW2,BW3},'Size',[1 5])
%     title("Normalized Image        |        Surpressed Image        |        Binary Image        |        BW with Euler Property        |        BW with Area Property ")
end