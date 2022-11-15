clear all
train = xlsread('DataForPerceptron.xlsx', 'TRAINData');
test = xlsread('DataForPerceptron.xlsx', 'TESTData');
train_x = train(:,2:10);
train_y = train(:,11);
test_x = test(:,2:10);
test_y = zeros(length(test_x),1);
length(test_x);

[predicted_class, weights] = prediction(train_x,train_y,0.1);

bias = ones(length(test_x),1); % Addition of Bias Feature
test_data = horzcat(test_x, bias);


for i = 1 : size(test_data,1)
    z = sum(weights.*test_data(i,:));
    % Activation Function
    if z > 2
        y_hat = 4;
    else
        y_hat = 2;
    end
    test_y(i) = y_hat;
end


for i = 1 : length(test_x)
    fprintf(' %d labels is %d \n',test(i,1),test_y(i))
end

sum_x = zeros(length(test_x),1);
for i=1 : length(test_x)
    sum_x(i) = sum(weights.*test_data(i,:));
end

%%%%%%%%% 
b = weights(1);
w1 = weights(2);
w2 = weights(3);
x = max(train_x(:,1));
for i = 1 : size(train_x,1)
    yline = -(b + w1*x)/w2;
    x = x - 0.1;
    line_x(i) = x;
    line_y(i) = yline;
end

figure('Name','Perceptron Classifier','NumberTitle','off')
xlim([0 11])
gscatter(test_x(:,1),sum_x,test_y);
grid on;
hold on
plot(line_x, line_y, 'linewidth', 2);
legend('Class1',' Class2','Decision boundary');
title('Decision boundary');

%%%%%%%%%%%%%%

function [y, w] = prediction(train_x,train_y, eta)

bias = ones(size(train_x,1),1); % Addition of Bias Feature
data = horzcat(train_x, bias);
[row, colm] = size(data);
w = zeros(colm,1);
w = transpose(w);

for k = 1 : 10000 % No of Iterations
    for i = 1 : row % For all Samples
        z = sum(w.* data(i,:)); % 
        % Activation Function
        if z > 2
            y_hat = 4;
        else
            y_hat = 2;
        end
        % Checking Weight
        delta = 0; 
        if train_y(i) ~= y_hat 
            delta = eta*(train_y(i) - y_hat).*data(i,:);
            w = w + delta;
        else 
            w = w;
        end
        y(i) = y_hat; 
    end  
end
end 
