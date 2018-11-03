mlpdata1 = csvread('mlpdata-1L.csv');
rbfdata2 = csvread('rbfdata-2S.csv');

rangemlp1 = mlpdata1(1,:);
minmlp1   = mlpdata1(2,:);
maxmlp1   = mlpdata1(3,:);
meanmlp1  = mlpdata1(4,:);
mintotalmlp1   = mlpdata1(5,:);
maxtotalmlp1   = mlpdata1(6,:);
meantotalmlp1  = mlpdata1(7,:);

rangerbf2 = rbfdata2(1,:);
minrbf2   = rbfdata2(2,:);
maxrbf2   = rbfdata2(3,:);
meanrbf2  = rbfdata2(4,:);
mintotalrbf2   = rbfdata2(5,:);
maxtotalrbf2   = rbfdata2(6,:);
meantotalrbf2  = rbfdata2(7,:);

hFig = figure(1);
set(hFig, 'Position', [450 200 1000 600])
axis tight

plot(rangemlp1, meanmlp1, 'b--o', 'MarkerFaceColor','b');
hold on
plot(rangerbf2, meanrbf2, 'r--o', 'MarkerFaceColor','r');

set(gca, 'FontSize', 16)
xlabel('Número de neurônios nas camadas ocultas', 'FontSize', 16);
ylabel('Taxa de acurácia no K-fold', 'FontSize', 16);
legend({'MLP com 1 camada oculta', 'RBF com raio receptivo r = 2'}, 'Location','southeast', 'FontSize', 16)
ylim([0 1])

saveTightFigure(hFig, 'mlp-rbf-graph.pdf')