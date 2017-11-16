function output = sigmoid(exponent)

output = 1./(1+exp(-exponent));
output(output<1e-16) = 1e-16;
output(output>1-(1e-16)) = 1-(1e-16);
    
end

