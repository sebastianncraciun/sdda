import numpy as np
import pandas as pd
import sklearn.cross_decomposition as skl
import matplotlib.pyplot as plt

t_ind = pd.read_csv('./dataIN/Industrie.csv', index_col=0)
t_pop = pd.read_csv('./dataIN/PopulatieLocalitati.csv', index_col=0)
#print(t_ind)
#print(t_pop)

industrii = list(t_ind.columns[1:].values)
print(industrii)

# Cerinta 1
t1 = t_ind.merge(right=t_pop, left_index=True, right_index=True)
t1.to_csv('./dataOUT/test_merged.csv')

def caLoc(t, variabile, populatie):
    x = t[variabile].values / t[populatie]
    #print(x)
    v = list(x)
    v.insert(0, t['Localitate_x'])
    return pd.Series(data=v, index=['Localitate'] + variabile)

# caLoc(t1[['Localitate_x', 'Populatie'] + industrii], industrii, 'Populatie')

t2 = t1[['Localitate_x', 'Populatie'] + industrii].apply(func=caLoc, axis=1,
                            variabile=industrii, populatie='Populatie')
print(t2)
t2.to_csv('./dataOUT/Cerinta1.csv')

# Cerinta 2
def maxCA(t):
    x = t.values
    max_linie = np.argmax(x)
    #print(max_linie)
    return pd.Series(data=[t.index[max_linie], x[max_linie]],
                     index=['Activitate', 'Cifra Afaceri'])


t3 = t1[industrii + ['Judet']].groupby(by='Judet').agg(sum)
print(t3)
t4 = t3[industrii].apply(func=maxCA, axis=1)
print(t4)
t4.to_csv('./dataOUT/Cerinta2.csv')

#Cerinta 3
tabel = pd.read_csv('./dataIN/DataSet_34.csv', index_col=0)
print(tabel)

obs_nume = tabel.index.values
var_nume = tabel.columns.values
X_coloane = var_nume[:4]
Y_coloane = var_nume[4:]

X = tabel[X_coloane].values
Y = tabel[Y_coloane].values

def standardizare(X):
    medii = np.mean(X, axis=0)
    abateri = np.std(X, axis=0)
    return (X - medii) / abateri

Xstd = standardizare(X)
Xstd_df = pd.DataFrame(data=Xstd, index=obs_nume, columns=X_coloane)
Xstd_df.to_csv('./dataOUT/Xstd.csv')
Ystd = standardizare(Y)
Ystd_df = pd.DataFrame(data=Xstd, index=obs_nume, columns=Y_coloane)
Ystd_df.to_csv('./dataOUT/Ystd.csv')


#Cerinta 4
#creare model ACC
n, p = np.shape(X)
q = np.shape(Y)[1] #nr de coloane
m = min(p, q)
print(n, p, q, m)

modelACC = skl.CCA(n_components=m)
modelACC.fit(X=Xstd, Y=Ystd)
z, u = modelACC.transform(X=Xstd, Y=Ystd)
#z = modelACC.x_scores_
print(z)
print(u)
z_df = pd.DataFrame(data=z, index=obs_nume,
                    columns=['z' + str(j+1) for j in range(p)])
z_df.to_csv('./dataOUT/Xscores.csv')
u_df = pd.DataFrame(data=u, index=obs_nume,
                    columns=['u' + str(j+1) for j in range(q)])
u_df.to_csv('./dataOUT/Yscores.csv')

#Cerinta 5
Rxz = modelACC.x_loadings_
Ryu = modelACC.y_loadings_

Rxz_df = pd.DataFrame(data=Rxz, index=X_coloane,
                      columns=['z' + str(j+1) for j in range(p)])
Rxz_df.to_csv('./dataOUT/Rxz.csv')

Ryu_df = pd.DataFrame(data=Ryu, index=Y_coloane,
                      columns=['u' + str(j+1) for j in range(q)])
Ryu_df.to_csv('./dataOUT/Ryu.csv')

#Cerinta 6
def biplot(x, y, xLabel='X', yLabel='Y', titlu='Bplot',
           e1=None, e2=None):
    f = plt.figure(figsize=(10, 7))
    ax = f.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontsize=14)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.scatter(x=x[:, 0], y=x[:, 1], c='r', label='Set X')
    ax.scatter(x=y[:, 0], y=y[:, 1], c='b', label='Set Y')
    if e1 is not None:
        for i in range (len(e1)):
            ax.text(x=x[i, 0], y=x[i, 1], s=e1[i])
    if e2 is not None:
        for i in range(len(e2)):
            ax.text(x=y[i, 0], y=y[i, 1], s=e2[i])

    ax.legend()

biplot(z[:, :2], u[:, :2],
       titlu='biplot tari in spatiul radacinilor unice',
       xLabel = '(z1, u1)', yLabel='(z2, u2)',
       e1=list(obs_nume), e2=list(obs_nume))
plt.show()