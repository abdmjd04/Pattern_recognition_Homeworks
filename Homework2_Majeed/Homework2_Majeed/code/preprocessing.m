function [ z, fmean, fstd ] = preprocessing(Data,type)





switch type
    case 1
        z=[];
        fmean = [];
        fstd = [];
        for i=1:length(Data(1,:))
            x = Data(:,i);
            z = [z (x-mean(x))./std(x)];
            fmean(:,i) = mean(x);
            fstd(:,i) = std(x);
        end
        
    case 2
        z=[];
        fmean = [];
        fstd = [];
        for i=1:length(Data(1,:))
            x=Data(:,i);
            z=[z log(x+0.1)];
            fmean(:,i) = mean(x);
            fstd(:,i) = std(x);
        end
    case 3
        z=[];
        fmean = [];
        fstd = [];
        for i=1:length(Data(1,:))
            x=Data(:,i);
            z=[z, x>0];
            fmean(:,i) = mean(x);
            fstd(:,i) = std(x);
        end
end

end
