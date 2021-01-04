from numpy import sum

def multiclass_metrics(confusion_matrix, metrics):
    tp = []
    tn = []
    fp = []
    fn = []
    total = sum(confusion_matrix)
    l = len(confusion_matrix)
    for i in range(l):
        tp.append(confusion_matrix[i, i])
        col = sum(confusion_matrix[:, i])
        row = sum(confusion_matrix[i])
        tn.append(total - col - row + tp[i])
        fp.append(col - tp[i])
        fn.append(row - tp[i])
    
    performance_metrics = []
    for metric in metrics:
        if metric == "avg_acc":
            avg_acc = 0
            for i in range(l):
                avg_acc += (tp[i]+tn[i])/(tp[i]+fn[i]+fp[i]+tn[i])
            avg_acc /= l
            performance_metrics.append(round(avg_acc*100, 3))
        elif metric == "err_rate":
            err_rate = 0
            for i in range(l):
                err_rate += (fp[i]+fn[i])/(tp[i]+fn[i]+fp[i]+tn[i])
            err_rate /= l
            performance_metrics.append(round(err_rate*100, 3))
        elif metric == "micro_prec":
            num = 0
            den = 0
            for i in range(l):
                num += tp[i]
                den += tp[i]+fp[i]
            micro_prec = num/den
            performance_metrics.append(round(micro_prec*100, 3))
        elif metric == "micro_recall":
            num = 0
            den = 0
            for i in range(l):
                num += tp[i]
                den += tp[i]+fn[i]
            micro_recall = num/den
            performance_metrics.append(round(micro_recall*100, 3))
        elif metric == "macro_prec":
            macro_prec = 0
            for i in range(l):
                macro_prec += tp[i]/(tp[i]+fp[i])
            macro_prec /= l
            performance_metrics.append(round(macro_prec*100, 3))
        elif metric == "macro_recall":
            macro_recall = 0
            for i in range(l):
                macro_recall += tp[i]/(tp[i]+fn[i])
            macro_recall /= l
            performance_metrics.append(round(macro_recall*100, 3))
        
    return performance_metrics


def binary_metrics(confusion_matrix, metrics):
    performance_metrics = []
    for metric in metrics:
        if metric == "acc":
            acc = (confusion_matrix[0,0]+confusion_matrix[1,1])/sum(confusion_matrix)
            performance_metrics.append(round(acc*100, 3))
        elif metric == "prec":
            prec = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
            performance_metrics.append(round(prec*100, 3))
        elif metric == "recall":
            recall = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
            performance_metrics.append(round(recall*100, 3))
        elif metric == "spec":
            spec = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
            performance_metrics.append(round(spec*100, 3))
        elif metric == "auc":
            auc = 0.5 * ((confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])) + (confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])))
            performance_metrics.append(round(auc*100, 3))
    
    return performance_metrics

