%%
clear variables;
close all;
clc;

%% Calling file

filename = 'CI_Assignment_HeartAttackPredictionDataset.xlsx'; 
opts = detectImportOptions(filename); 
opts.VariableNamingRule = 'preserve'; 
data = readtable(filename, opts);

%% Initialization

Age = data{:, 2};
Sex = categorical(data{:, 3});
Cholesterol = data{:, 4};

%% Converting blood pressure to MAP

BloodPressure = data{:, 5}; 
bp_values = BloodPressure;

map_values = zeros(length(BloodPressure), 1);% Create array of zeros

for i = 1:length(bp_values) % Get the current blood pressure reading
    current_bp = bp_values{i};
    
    % Split the string into systolic and diastolic components
    bp_split = strsplit(current_bp, '/');
    systolic = str2double(bp_split{1});
    diastolic = str2double(bp_split{2});
    
    % Calculate Mean Arterial Pressure (MAP)
    map_values(i) = (2 * diastolic + systolic) / 3;
    
end

%% Continue initialization

HeartRate = data{:, 6};
Diabetes = data{:, 7};
FamilyHistory = data{:, 8};
Smoking = data{:, 9};
Obesity = data{:,10};
Alcohol = data{:,11};
Exercise = data{:,12};
Diet = categorical(data{:, 13});
PreviousHeartProblems = data{:,14};
MedicationUse = data{:,15};
StressLevel = data{:,16};
SedentaryHours = data{:,17};
Income = data{:,18};
BMI = data{:,19};
Triglycerides = data{:,20};
PhysicalActivity = data{:,21};
SleepHours = data{:,22};
Country = categorical(data{:,23});
Continent = categorical(data{:,24});
Hemisphere = categorical(data{:,25});


%% One Hot Encoding

oneHotEncodedSex = onehotencode(Sex, 2);
oneHotEncodedDiet = onehotencode(Diet, 2);
oneHotEncodedCountry = onehotencode(Country, 2);
oneHotEncodedContinent = onehotencode(Continent, 2);
oneHotEncodedHemisphere = onehotencode(Hemisphere, 2);

%% Input matrix

input = [Age oneHotEncodedSex Cholesterol map_values...
    HeartRate Diabetes FamilyHistory Smoking Obesity Alcohol...
    Exercise oneHotEncodedDiet PreviousHeartProblems MedicationUse...
    StressLevel SedentaryHours Income BMI Triglycerides...
    oneHotEncodedCountry oneHotEncodedContinent oneHotEncodedHemisphere];

%% Output matrix

output = data{:, 26}; % Heart disease risk

%% Training

% Reinitialize for clarity
% Convert input and output to double
% Normalize and check for outliers
inputData = normalize(input, 'range'); % Normalize all inputs to [0, 1]
outputData = double(output);

% Visualize data using a boxplot for potential outliers
figure;
boxplot(inputData);
title('Input Data Boxplot');
xlabel('Features');
ylabel('Normalized Range');

% Step 1: Split Data (70% training, 30% testing)
cv = cvpartition(size(inputData, 1), 'HoldOut', 0.3); % *% for testing
trainIdx = ~cv.test; % Indices for training data
testIdx = cv.test;   % Indices for testing data

inputTrain = inputData(trainIdx, :);
outputTrain = outputData(trainIdx, :);
inputTest = inputData(testIdx, :);
outputTest = outputData(testIdx, :);

% Step 2: Reduce Input Dimensionality (PCA on training data)
disp('Applying PCA...');
[coeff, score, ~] = pca(inputTrain);
numComponents = 5; % *Reduce to fewer components to avoid memory overload
inputTrainReduced = score(:, 1:numComponents);
inputTestReduced = (inputTest - mean(inputTrain)) * coeff(:, ...
    1:numComponents);

% Step 3: Subset Training Data for FIS Generation
disp('Subsetting training data for FIS generation...');
numSamples = 500; % *Reduce samples for clustering
if size(inputTrainReduced, 1) > numSamples
    idx = randperm(size(inputTrainReduced, 1), numSamples);
else
    idx = 1:size(inputTrainReduced, 1);
end
inputSubset = inputTrainReduced(idx, :);
outputSubset = outputTrain(idx, :);

% Step 4: Generate FIS with Subtractive Clustering
disp('Generating FIS...');
options = genfisOptions('SubtractiveClustering');
options.ClusterInfluenceRange = 0.3; % *
fis = genfis(inputSubset, outputSubset, options);

% Step 5: Train FIS using ANFIS
disp('Training FIS...');
trainingData = [inputTrainReduced, outputTrain];
anfisOptions = anfisOptions('InitialFIS', fis, 'EpochNumber', 10,... %*
    'DisplayANFISInformation', 0, 'DisplayErrorValues', 1);
[trainedFis, trainingError] = anfis(trainingData, anfisOptions);

save('CI_Assignment_trainedFIS.mat', 'trainedFis'); % Saving trained file

%% Testing

load('CI_Assignment_trainedFIS.mat', 'trainedFis'); % Loading trained file

% Step 6: Testing and Prediction
disp('Testing the trained FIS...');
predictedOutput = evalfis(trainedFis, inputTestReduced); % Use reduced test data

% Calculating error
error = outputTest - predictedOutput;
RMSE = sqrt(mean(error.^2));
fprintf('Root Mean Square Error (RMSE): %.4f\n', RMSE);

% Evaluate performance
figure;
disp('Plotting performance...');
plot(outputTest, 'b'); % Actual
hold on;
plot(predictedOutput, 'r--'); % Prediction
legend('Actual Output', 'Predicted Output');
title('Comparison of Actual and Predicted Outputs');
xlabel('Samples');
ylabel('Heart Attack Risk');

% Optional: Plot ANFIS training error
figure;
plot(trainingError);
title('ANFIS Training Error');
xlabel('Epochs');
ylabel('Error');

% Change predicted output to discrete values for classification
predictedClasses = round(predictedOutput);

% Confusion matrix
confusionchart(outputTest, predictedClasses);

% Calculate accuracy
accuracy = sum(predictedClasses == outputTest) / length(outputTest) * 100;
fprintf('Accuracy: %.2f%%\n', accuracy);








