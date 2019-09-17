%% Import Data.
fileNum = [];
if ~isempty(fileNum)
    fileNume = ['(', fileNum, ')'];
end
dc	= dlmread(['~/Documents/IE6380_Project/Results/dice_coef', fileNum]);
jc	= dlmread(['~/Documents/IE6380_Project/Results/jaccard_coef', fileNum]);
vloss	= dlmread(['~/Documents/IE6380_Project/Results/val_loss', fileNum]);
% acc	= dlmread(['~/Documents/IE6380_Project/Results/acc', fileNum]);
epochs  = 1:length(dc);

%% Plot Figure.
orange = [0.9100 0.4100 0.1700];
f = figure('units', 'normalized', 'OuterPosition', [0 0 1 1], 'Color', 'w');
hold on;
yyaxis left
ylabel('Loss Value');
plot(epochs, dc, 'ro-', 'linewidth', 2.5, 'markerfacecolor', 'r');
plot(epochs, jc, 'r--', 'linewidth', 2.5);
set(gca, 'YColor', 'r', 'YLim', [0 1.0], 'YTick', 0:.1:1.0);
yyaxis right
ylabel('Loss Value');
plot(epochs, vloss, 'mo-', 'linewidth', 2.5, 'markerfacecolor', 'm')
set(gca, 'YColor', 'm', 'YLim', [-1.0 0], 'YTick', -1:0.1:0);
legend('Dice Coefficient', 'Jaccard Coefficient',...
    'Validation Loss', 'location', 'best')
title(['Model Validation Over ', num2str(epochs(end)), ' Epochs'])
xlabel('Epochs');
set(gca, 'XLim', [0 50], 'XTick', 0:5:50);
set(gca,'FontSize', 18, 'fontweight', 'bold');
box on;
grid on;
grid minor