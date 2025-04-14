from data import (
    perform_gmm_clustering,
    read_preprocessed_data,
    apply_kmeans,
    apply_pca,
    summarize_clusters,
    plot_cluster_profiles,
    plot_clusters,
    print_pca_variance,
optimal_jaccard
)
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")


results = {
    "GMM":{},
    "K-means":{}
}
result_summaries = {
    "GMM":{},
    "K-means":{}
}

result_PCAS = {
    "GMM":{},
    "K-means":{}
}
met1= ["GMM",'K-means']
met2=["Standaryzacja","MinMax","Robust"]
#column and row
cow = []
for i in range(len(met1)):
    for n in range(len(met2)):
        cow.append(met1[i]+"_"+met2[n])

comparison_results = pd.DataFrame(-100.0
                  , columns=cow, index=cow)
comparison_replacements = pd.DataFrame(" "
                  , columns=cow, index=cow)




#win2


data= read_preprocessed_data(gmm=True)
data, gmm_model = perform_gmm_clustering(n_clusters=4, data_scaled=data)
results["GMM"]["Standaryzacja"]=data
data, pca = apply_pca(data)
result_PCAS["GMM"]["Standaryzacja"]=round(pca.explained_variance_ratio_.sum(), 2)

data = read_preprocessed_data(std=False, minmax=True,gmm=True)
data, gmm_model = perform_gmm_clustering(n_clusters=4, data_scaled=data)
results["GMM"]["MinMax"]=data
data, pca = apply_pca(data)
result_PCAS["GMM"]["MinMax"]=round(pca.explained_variance_ratio_.sum(), 2)


data = read_preprocessed_data(std=False, robust=True,gmm=True)
data, gmm_model = perform_gmm_clustering(n_clusters=4, data_scaled=data)
results["GMM"]["Robust"]=data
data, pca = apply_pca(data)
result_PCAS["GMM"]["Robust"]=round(pca.explained_variance_ratio_.sum(), 2)

data= read_preprocessed_data(le=False, he=True)
data, gmm_model = perform_gmm_clustering(n_clusters=4, data_scaled=data)
results["K-means"]["Standaryzacja"]=data
data, pca = apply_pca(data)
result_PCAS["K-means"]["Standaryzacja"]=round(pca.explained_variance_ratio_.sum(), 2)



data = read_preprocessed_data(std=False, minmax=True, le=False, he=True)
data, gmm_model = perform_gmm_clustering(n_clusters=4, data_scaled=data)
results["K-means"]["MinMax"]=data
data, pca = apply_pca(data)
result_PCAS["K-means"]["MinMax"]=round(pca.explained_variance_ratio_.sum(), 2)




data = read_preprocessed_data(std=False, robust=True, le=False, he=True)
data, gmm_model = perform_gmm_clustering(n_clusters=4, data_scaled=data)
results["K-means"]["Robust"]=data
data, pca = apply_pca(data)
result_PCAS["K-means"]["Robust"]=round(pca.explained_variance_ratio_.sum(), 2)



for i in range(len(met1)):
    for j in range(len(met2)):
        l=i
        for l in range(len(met1)):
            k= j if i==l else 0
            for k in range(len(met2)):
                comparison_results.loc[met1[i]+"_"+met2[j],met1[l]+"_"+met2[k]]= optimal_jaccard(results[met1[i]][met2[j]],results[met1[l]][met2[k]])
                comparison_results.loc[met1[l]+"_"+met2[k],met1[i]+"_"+met2[j]]= optimal_jaccard(results[met1[i]][met2[j]],results[met1[l]][met2[k]])

print("tabela poziomu podobieństwa wyników klasteryzacji różnymi metodami danych skalowanych różnymi metodami")
print(comparison_results.to_string())


print("Wariancje PCA")
for val1 in met1:
    for val2 in met2:
        print( val1 + "_"+val2+": " +str(result_PCAS[val1][val2]))






for i in range(len(met1)):
    for j in range(len(met2)):
        l=i
        for l in range(len(met1)):
            k= j if i==l else 0
            for k in range(len(met2)):
                comparison_replacements.loc[met1[i]+"_"+met2[j],met1[l]+"_"+met2[k]]= optimal_jaccard(results[met1[i]][met2[j]],results[met1[l]][met2[k]],result=False)
print("Tabela pokazująca dopasowanie klastrów z jednych wyników (rzędy) do klastrów z drugich wyników (kolumny)")
print(comparison_replacements.to_string())



results["GMM"]["MinMax"].loc[results["K-means"]["Robust"]["Cluster"] == 1,"Cluster"] = -100
results["GMM"]["MinMax"].loc[results["K-means"]["Robust"]["Cluster"] == 0,"Cluster"] = -101

results["K-means"]["MinMax"].loc[results["K-means"]["Robust"]["Cluster"] == 1,"Cluster"] = -100
results["K-means"]["MinMax"].loc[results["K-means"]["Robust"]["Cluster"] == 0,"Cluster"] = -101



results["GMM"]["MinMax"].loc[results["K-means"]["Robust"]["Cluster"] == -100,"Cluster"] = 0
results["GMM"]["MinMax"].loc[results["K-means"]["Robust"]["Cluster"] == -101,"Cluster"] = 1

results["K-means"]["MinMax"].loc[results["K-means"]["Robust"]["Cluster"] == -100,"Cluster"] = 0
results["K-means"]["MinMax"].loc[results["K-means"]["Robust"]["Cluster"] == -101,"Cluster"] = 1
#dla 'translacji' klastrów minmax na kalstry standaryzacji  
#2=2,3=3, jedynie 0=1 i 1=0

results["GMM"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == 1,"Cluster"] = -100
results["GMM"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == 3,"Cluster"] = -101
results["GMM"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == 0,"Cluster"] = -102
results["GMM"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == 2,"Cluster"] = -103



results["K-means"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == 1,"Cluster"] = -100
results["K-means"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == 3,"Cluster"] = -101
results["K-means"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == 0,"Cluster"] = -102
results["K-means"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == 2,"Cluster"] = -103



results["GMM"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == -100,"Cluster"] = 0
results["GMM"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == -101,"Cluster"] = 1
results["GMM"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == -102,"Cluster"] = 2
results["GMM"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == -103,"Cluster"] = 3



results["K-means"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == -100,"Cluster"] = 0
results["K-means"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == -101,"Cluster"] = 1
results["K-means"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == -102,"Cluster"] = 2
results["K-means"]["Robust"].loc[results["K-means"]["Robust"]["Cluster"] == -103,"Cluster"] = 3






for i in range(len(met1)):
    for j in range(len(met2)):
        original_data = read_preprocessed_data(std=False, le=False)
        original_data["Cluster"] = results[met1[i]][met2[j]]["Cluster"]
        cluster_summary = original_data.groupby("Cluster").mean(numeric_only=True)
        object_cols = original_data.select_dtypes(include="object").columns
        for col in object_cols:
            mode_per_cluster = original_data.groupby("Cluster")[col].agg(lambda x: x.mode()[0])
            cluster_summary[col + "_mode"] = mode_per_cluster
        
        original_data["Spending_To_Income_Ratio"] = original_data["Spent"]/original_data["Income"]
        cluster_summary["Spending_To_Income_Ratio_Mode"] = original_data.groupby("Cluster")["Spending_To_Income_Ratio"].agg(lambda x: x.mode()[0])
        
        cluster_summary["Count"] = original_data["Cluster"].value_counts().sort_index()
        result_summaries[met1[i]][met2[j]]=cluster_summary.transpose()






row_names = result_summaries["GMM"]["Standaryzacja"][0].index.tolist()
result_summaries["GMM"]["Standaryzacja"][0].head()
cl=[]

cl.append( pd.DataFrame(columns=cow, index=row_names))
for i in range(len(met1)):
    for j in range(len(met2)):
         cl[0][met1[i]+"_"+met2[j]]= result_summaries[met1[i]][met2[j]][0]
print("Porównanie klastrów typu '0' ze wszystkich wyników")
print(cl[0].to_string())



cl.append(pd.DataFrame(columns=cow, index=row_names))
for i in range(len(met1)):
    for j in range(len(met2)):
         cl[1][met1[i]+"_"+met2[j]]= result_summaries[met1[i]][met2[j]][1]
print("Porównanie klastrów typu '1' ze wszystkich wyników")
print(cl[1].to_string())


cl.append(pd.DataFrame(columns=cow, index=row_names))
for i in range(len(met1)):
    for j in range(len(met2)):
         cl[2][met1[i]+"_"+met2[j]]= result_summaries[met1[i]][met2[j]][2]
print("Porównanie klastrów typu '2' ze wszystkich wyników")
print(cl[2].to_string())



cl.append(pd.DataFrame(columns=cow, index=row_names))
for i in range(len(met1)):
    for j in range(len(met2)):
         cl[3][met1[i]+"_"+met2[j]]= result_summaries[met1[i]][met2[j]][3]
print("Porównanie klastrów typu '3' ze wszystkich wyników")
print(cl[3].to_string())


cluster_summaries=pd.DataFrame(columns=[0,1,2,3],index=cl[0].transpose().columns.drop("Marital_Status_mode").drop("Education_mode"))
for n in range(4):
    cs=cl[n].transpose()
    
    cluster_summary=pd.DataFrame(columns=cs.columns,index=[0])
    for c in cs.columns:
        if(isinstance(cs[c][0],float) and c!="Count"):
            temp=0
            count = 0
            for i in range(len(cs[c])):
                if(str(cs[c][i])!='nan'):
                    temp+=cs[c][i]*cs['Count'][i]
                    count+= cs['Count'][i]
            cluster_summary.loc[0,c]=temp/count
    cluster_summary.loc[0,"Count"]=count/6
    cluster_summaries[n]=cluster_summary.transpose()    
                    
print("Średnie z wartości odpowiadających sobie klastrów ze wszystkich metod")
print(cluster_summaries.to_string())

      