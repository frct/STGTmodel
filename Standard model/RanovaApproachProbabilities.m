% 2 factor repeated measures ANOVA on the number of goL and goM choices

%% prologue

clear all
close all

%% load data

load('goL_counter for Short ITI.mat')
[n_rats, n_sessions] = size(goL);
GoL = goL;
load('goL_counter for Long ITI.mat')
GoL = [GoL, goL];

load('goM_counter for Short ITI.mat')
[n_rats, n_sessions] = size(goM);
GoM = goM;
load('goM_counter for Long ITI.mat')
GoM = [GoM, goM];

%% initialize variable and within factor tables for fitrm function

ITI = {'short'; 'short'; 'short'; 'short'; 'short'; 'short'; 'short'; 'short'; 'short'; 'short'; ...
    'long'; 'long'; 'long'; 'long'; 'long'; 'long'; 'long'; 'long'; 'long'; 'long'};

goL_table = table;
goM_table = table;

goL_table.subjects = [1 : n_rats]';
goM_table.subjects = [1 : n_rats]';

j = 1;

for s = 1 : 2 * n_sessions
    col_name = ['col' num2str(j)];
    goL_table.(col_name) = GoL(:, s);
    goM_table.(col_name) = GoM(:, s);
    j = j + 1;
end


within = table(repmat([1 : n_sessions]', 2, 1), ITI, 'VariableNames', {'sessions', 'ITIs'});

%% testing gol

rm_goL = fitrm(goL_table,'col1-col20~1', 'WithinDesign', within)
mauchly(rm_goL)
epsilon(rm_goL)
tbl = ranova(rm_goL, 'WithinModel', 'sessions*ITIs')

posthoc_ttest_goL = multcompare(rm_goL, 'ITIs', 'By', 'sessions', 'ComparisonType', 'Bonferroni')

%% testing goM

rm_goM = fitrm(goM_table,'col1-col20~1', 'WithinDesign', within)
mauchly(rm_goM)
epsilon(rm_goM)
tbl = ranova(rm_goM, 'WithinModel', 'sessions*ITIs')

posthoc_ttest_goM = multcompare(rm_goM, 'ITIs', 'By', 'sessions', 'ComparisonType', 'Bonferroni')