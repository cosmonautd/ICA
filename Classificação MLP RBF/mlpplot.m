mlpdata1 = csvread('mlpdata-1L.csv');
mlpdata2 = csvread('mlpdata-2L.csv');

rangemlp1 = mlpdata1(1,:);
minmlp1   = mlpdata1(2,:);
maxmlp1   = mlpdata1(3,:);
meanmlp1  = mlpdata1(4,:);
mintotalmlp1   = mlpdata1(5,:);
maxtotalmlp1   = mlpdata1(6,:);
meantotalmlp1  = mlpdata1(7,:);

rangemlp2 = mlpdata2(1,:);
minmlp2   = mlpdata2(2,:);
maxmlp2   = mlpdata2(3,:);
meanmlp2  = mlpdata2(4,:);
mintotalmlp2   = mlpdata2(5,:);
maxtotalmlp2   = mlpdata2(6,:);
meantotalmlp2  = mlpdata2(7,:);

[m1 i1] = max(meanmlp1);
[m2 i2] = max(meanmlp2);

best1 = rangemlp1(i1);
best2 = rangemlp2(i2);

if m1 > m2
    neurons = best1;
    fprintf('Best architecture has 1 layer and %d neurons. K-fold result: %.4f\n', neurons, m1) ;
else
    neurons = best2;
    fprintf('Best architecture has 2 layers and %d neurons. K-fold result: %.4f\n', neurons, m2) ;
end

hFig = figure(1);
set(hFig, 'Position', [450 200 1000 600])
axis tight

plot(rangemlp1, meanmlp1, 'b--o', 'MarkerFaceColor','b');
hold on
plot(rangemlp2, meanmlp2, 'r--o', 'MarkerFaceColor','r');

xlabel('Número de neurônios nas camadas ocultas');
ylabel('Taxa de acerto média K-fold');
legend('1 camada oculta', '2 camadas ocultas', 'Location','southeast')
ylim([0 1])

saveTightFigure(hFig, 'mlp-graph.pdf')