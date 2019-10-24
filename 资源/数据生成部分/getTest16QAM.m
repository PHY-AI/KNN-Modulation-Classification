%% Clean up
clear all;
clc;

rows = 1000;
cols = 500*4;
NFeatures = 9;
test_data = zeros(rows, NFeatures);
SNR = [0:5:40];
for i = 1:length(SNR)
    
%% Simulation parameters
M = 16;
snr = SNR(i);

%% Simulate
    %16QAM
    % Generate random data
    data = randi([0 1], rows, cols);   
    
    % QAM ӳ��
    [data1] = data';
    data_cell = mat2cell(data1,cols,ones(1,rows));
    signalData = cellfun(@(x) qammod(x, M, 'InputType', 'bit', 'UnitAveragePower', true), data_cell, 'UniformOutput',false);
    signalDataMat = cell2mat(signalData);
    dataMod = mat2cell(signalDataMat',ones(1,rows),cols/4);
    
     % AWGN ����
    dataRx = cellfun(@(x) awgn(x, snr), dataMod, 'UniformOutput',false);
    
    % ��ȡ�߽��ۻ���
    cumulants = cellfun(@(x) func_get_cumulants(x), dataRx, 'UniformOutput',false);
    cumulantsMat = cell2mat(cumulants);
    
    for row = 1: rows
        for i = 1:NFeatures
                test_data(row, i) = cumulantsMat(row, i);
        end
    end 

%% save
filename=['digits\test16QAM-',num2str(snr),'.dat'];
dlmwrite(filename,test_data,'delimiter','\t','newline','pc');
end