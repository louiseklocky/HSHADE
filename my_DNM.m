function Q=my_DNM(data,net)
M = net.M;
qs = net.qs;
k = net.k;
w = net.w;
q = net.q;
P=data';
[S,J]=size(P); % J is the number of samples; I is the dimension of sample.
b=round((S-1)/2)-2;
Q=zeros(J,1);
for h=1:J
    Train_in2=repmat(P(:,h),1,M);
    Y=1./(1+exp(-k*(w.*Train_in2-q)));
    Z=prod(Y,1);
    V=sum(Z);
    V=(10^b)*V;
    O=1./(1+exp(-k*(V-qs)));
    Q(h) = O;
end
end


