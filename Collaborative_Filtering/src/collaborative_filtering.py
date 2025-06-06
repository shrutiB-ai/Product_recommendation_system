
from surprise import KNNBasic,SVD,accuracy
from surprise.model_selection import train_test_split

def get_algorithm(algo_name,params):
    if algo_name.lower()=='svd':
        return SVD(
            n_factors=params.get('n_factors',100),
            n_epochs=params.get('n_epochs',20),
            lr_all = params.get('lr_all',0.005),
            reg_all = params.get('reg_all',0.02))
    elif algo_name.lower()=='knnbasic':
        sim_options = params.get('sim_options',{'name':'cosine','user_based':True})
        k = params.get('k',40)
        return KNNBasic(k=k,sim_options=sim_options)
    else:
        raise ValueError(f'Algorithm {algo_name} not supported')

def run_surprise_cf(data, algo_name,params):
    train_set,test_set = train_test_split(data,test_size=0.25,random_state=42)
    algo= get_algorithm(algo_name,params)
    algo.fit(train_set)
    predictions = algo.test(test_set)
    
    rmse = accuracy.rmse(predictions,verbose=False)
    mae= accuracy.mae(predictions,verbose=False)

    results = {
        'rmse':rmse,
        'mae':mae,
        'predictions':predictions}
    return results

  
def get_top_recommendations(algo, data , n=5):
    train_set = data.build_full_trainset()
    all_user_inner_ids=train_set.all_users()
    all_item_inner_ids=train_set.all_items()
    all_items = [train_set.to_raw_iid(iid) for iid in all_item_inner_ids]
    top_n = {}

    for uid_inner in all_user_inner_ids:
        uid = train_set.to_raw_uid(uid_inner)
        user_rated_items = set([train_set.to_raw_iid(iid) for (iid, _) in train_set.ur[uid_inner]])
        items_to_pred = [iid for iid in all_items if iid not in user_rated_items]
        predictions = [algo.predict(uid,iid) for iid in items_to_pred]
        predictions.sort(key=lambda x:x.est,reverse=True)
        top_n[uid] = [(pred.iid,round(pred.est,3)) for pred in predictions[:n]]
    return top_n