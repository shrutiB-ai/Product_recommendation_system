import yaml 
import logging
from src.data_loader import load_interactions
from surprise import Dataset,Reader
from src.collaborative_filtering import run_suprise_cf, get_top_recommendations
from src.visualization import plot_comparison_metrics
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s' )

def load_config(path='config.yaml'):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def main():
    config=load_config()
    data_path = config['data']['path']
    output_dir = config['output']['dir']
    n_recommendation = config['recommendation'].get('top_n',5)

    data = load_interactions(data_path)
    algo_results = {}
    for algo_name in config['model']['types']:
        params = config['model']['params'][algo_name]
        logging.info(f'Running algorith {algo_name}')
        results = run_suprise_cf(data,algo_name,params)
        algo_results[algo_name]=results
        top_n = get_top_recommendations(results['algo'],data,n=n_recommendation)
        with open(f"{output_dir}/top_{n_recommendation}_recommendation_{algo_name}.txt","w") as f:
            for user_id,recs in top_n.items():
                f.write(f"User {user_id} recommendations : {recs}\n")

    plot_comparison_metrics(algo_results,output_dir)
if __name__=="__main__":
    main()