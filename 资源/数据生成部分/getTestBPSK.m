%% Clean up
clear all;
clc;

rows = 1000;
cols = 500;
NFeatures = 9;
test_data = zeros(rows, NFeatures);
SNR = [-4:2:10];
for i = 1:length(SNR)
    
%% Simulation parameters
M = 2;
snr = SNR(i);
%% Simulate
    %BPSK
    % Generate random data
    data = randi([0 1], rows, cols);   
    
    % PSK ӳ��
    [data1] = data';
    data_cell = mat2cell(data1,cols,ones(1,rows));
    signalData = cellfun(@(x) BPSKModulator(x), data_cell, 'UniformOutput',false);
    signalDataMat = cell2mat(signalData);
    dataMod = mat2cell(signalDataMat',ones(1,rows),cols);
    
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
filename=['digits\testBPSK-',num2str(snr),'.dat'];
dlmwrite(filename,test_data,'delimiter','\t','newline','pc');
end