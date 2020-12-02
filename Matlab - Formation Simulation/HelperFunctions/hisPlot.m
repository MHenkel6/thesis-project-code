function [graph1,graph2] = hisPlot(data,faultIndex,distToPlot)
if faultIndex > 1
    graph1 = figure;
    sgtitle("Distribution before fault")
    subplot(2,3,1)
    histogram(data(1:faultIndex,(distToPlot-1)*6+1),'Normalization', 'pdf' )
    hold on
    [m,s] = normfit(data(1:faultIndex,(distToPlot-1)*6+1));
    y = normpdf(data(1:faultIndex,(distToPlot-1)*6+1),m,s);
    plot(data(1:faultIndex,(distToPlot-1)*6+1),y,'.')

    subplot(2,3,2)
    histogram(data(1:faultIndex,(distToPlot-1)*6+2),'Normalization', 'pdf' )
    hold on
    [m,s] = normfit(data(1:faultIndex,(distToPlot-1)*6+2));
    y = normpdf(data(1:faultIndex,(distToPlot-1)*6+2),m,s);
    plot(data(1:faultIndex,(distToPlot-1)*6+2),y,'.')

    subplot(2,3,3)
    histogram(data(1:faultIndex,(distToPlot-1)*6+3),'Normalization', 'pdf' )
    hold on
    [m,s] = normfit(data(1:faultIndex,(distToPlot-1)*6+3));
    y = normpdf(data(1:faultIndex,(distToPlot-1)*6+3),m,s);
    plot(data(1:faultIndex,(distToPlot-1)*6+3),y,'.')

    subplot(2,3,4)
    histogram(data(1:faultIndex,(distToPlot-1)*6+4),'Normalization', 'pdf' )
    hold on
    [m,s] = normfit(data(1:faultIndex,(distToPlot-1)*6+4));
    y = normpdf(data(1:faultIndex,(distToPlot-1)*6+4),m,s);
    plot(data(1:faultIndex,(distToPlot-1)*6+4),y,'.')

    subplot(2,3,5)
    histogram(data(1:faultIndex,(distToPlot-1)*6+5),'Normalization', 'pdf' )
    hold on
    [m,s] = normfit(data(1:faultIndex,(distToPlot-1)*6+5));
    y = normpdf(data(1:faultIndex,(distToPlot-1)*6+5),m,s);
    plot(data(1:faultIndex,(distToPlot-1)*6+5),y,'.')

    subplot(2,3,6)
    histogram(data(1:faultIndex,(distToPlot-1)*6+6),'Normalization', 'pdf' )
    hold on
    [m,s] = normfit(data(1:faultIndex,(distToPlot-1)*6+6));
    y = normpdf(data(1:faultIndex,(distToPlot-1)*6+6),m,s);
    plot(data(1:faultIndex,(distToPlot-1)*6+6),y,'.')
else
    faultIndex = 1;
end
% Second graph: after fault
graph2 = figure;
sgtitle("Distribution after Fault")
subplot(2,3,1)
histogram(data(faultIndex:end,(distToPlot-1)*6+1),'Normalization', 'pdf' )
hold on
[m,s] = normfit(data(faultIndex:end,(distToPlot-1)*6+1));
y = normpdf(data(faultIndex:end,(distToPlot-1)*6+1),m,s);
plot(data(faultIndex:end,(distToPlot-1)*6+1),y,'.')

subplot(2,3,2)
histogram(data(faultIndex:end,(distToPlot-1)*6+2),'Normalization', 'pdf' )
hold on
[m,s] = normfit(data(faultIndex:end,(distToPlot-1)*6+2));
y = normpdf(data(faultIndex:end,(distToPlot-1)*6+2),m,s);
plot(data(faultIndex:end,(distToPlot-1)*6+2),y,'.')

subplot(2,3,3)
histogram(data(faultIndex:end,(distToPlot-1)*6+3),'Normalization', 'pdf' )
hold on
[m,s] = normfit(data(faultIndex:end,(distToPlot-1)*6+3));
y = normpdf(data(faultIndex:end,(distToPlot-1)*6+3),m,s);
plot(data(faultIndex:end,(distToPlot-1)*6+3),y,'.')

subplot(2,3,4)
histogram(data(faultIndex:end,(distToPlot-1)*6+4),'Normalization', 'pdf' )
hold on
[m,s] = normfit(data(faultIndex:end,(distToPlot-1)*6+4));
y = normpdf(data(faultIndex:end,(distToPlot-1)*6+4),m,s);
plot(data(faultIndex:end,(distToPlot-1)*6+4),y,'.')

subplot(2,3,5)
histogram(data(faultIndex:end,(distToPlot-1)*6+5),'Normalization', 'pdf' )
hold on
[m,s] = normfit(data(faultIndex:end,(distToPlot-1)*6+5));
y = normpdf(data(faultIndex:end,(distToPlot-1)*6+5),m,s);
plot(data(faultIndex:end,(distToPlot-1)*6+5),y,'.')

subplot(2,3,6)
histogram(data(faultIndex:end,(distToPlot-1)*6+6),'Normalization', 'pdf' )
hold on
[m,s] = normfit(data(faultIndex:end,(distToPlot-1)*6+6));
y = normpdf(data(faultIndex:end,(distToPlot-1)*6+6),m,s);
plot(data(faultIndex:end,(distToPlot-1)*6+6),y,'.')
end