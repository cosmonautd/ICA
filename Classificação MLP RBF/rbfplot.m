rbfdata1 = csvread('rbfdata-1S.csv');
rbfdata2 = csvread('rbfdata-2S.csv');

rangerbf1 = rbfdata1(1,:);
minrbf1   = rbfdata1(2,:);
maxrbf1   = rbfdata1(3,:);
meanrbf1  = rbfdata1(4,:);
mintotalrbf1   = rbfdata1(5,:);
maxtotalrbf1   = rbfdata1(6,:);
meantotalrbf1  = rbfdata1(7,:);

rangerbf2 = rbfdata2(1,:);
minrbf2   = rbfdata2(2,:);
maxrbf2   = rbfdata2(3,:);
meanrbf2  = rbfdata2(4,:);
mintotalrbf2   = rbfdata2(5,:);
maxtotalrbf2   = rbfdata2(6,:);
meantotalrbf2  = rbfdata2(7,:);

[m1 i1] = max(meanrbf1);
[m2 i2] = max(meanrbf2);

best1 = rangerbf1(i1);
best2 = rangerbf2(i2);

if m1 > m2
    neurons = best1;
    fprintf('Best architecture has %d neurons and spread = 1.0. K-fold result: %.4f\n', neurons, m1);
else
    neurons = best2;
    fprintf('Best architecture has %d neurons and spread = 2.0. K-fold result: %.4f\n', neurons, m2);
end

hFig = figure(1);
set(hFig, 'Position', [450 200 1000 600])
axis tight

plot(rangerbf1, meanrbf1, 'b--o', 'MarkerFaceColor','b');
hold on
plot(rangerbf2, meanrbf2, 'r--o', 'MarkerFaceColor','r');

xlabel('Número de neurônios nas camadas ocultas');
ylabel('Taxa de acerto média K-fold');
legend('RBF com raio receptivo = 1.0', 'RBF com raio receptivo = 2.0', 'Location','southeast')
ylim([0 1])

saveTightFigure(hFig, 'rbf-graph.pdf')