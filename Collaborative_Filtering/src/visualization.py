import matplotlib.pyplot as plt
import os

def plot_comparison_metrics(results_dict,output_dir):
    algos = list(results_dict.keys())
    rmses = [results_dict[a]['rmse']for a in algos]
    maes = [results_dict[a]['mae'] for a in algos]

    x=range(len(algos))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.bar(x,rmses,coor='skyblue')
    plt.xticks(x,algos)
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')

    plt.subplot(1,2,2)
    plt.bar(x,maes,color='salmon')
    plt.xticks(x,algos)
    plt.title('MAE Comparison')
    plt.ylabel('MAE')

    plt.tight_layout()
    os.makedirs(output_dir,exist_ok=True)
    save_path=os.path.join(output_dir,"model_comparision_metrics.png")
    plt.savefig(save_path)
    plt.close()
    plt.show()