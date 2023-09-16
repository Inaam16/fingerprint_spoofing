import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

data = []
with open('GMM_results_pca6.txt') as file:
    for line in file:
        data.append(line.strip().split(','))
dfp = pd.DataFrame(data, columns=["components_0", 'components_1', 'diag0', 'tied0', 'diag1', 'tied1', 'minDCF'])

data = []
with open('GMM_results_pca0.txt') as file:
    for line in file:
        data.append(list(map(lambda x: x.strip(), line.strip().split(','))))
df = pd.DataFrame(data, columns=["components_0", 'components_1', 'diag0', 'tied0', 'diag1', 'tied1', 'minDCF'])


for i in ['diag0', 'tied0', 'diag1', 'tied1']:
    dfp[i] = dfp[i] == 'True'
    df[i] = df[i] == 'True'

dfp = dfp.astype({"components_0": int, 'components_1': int, 'diag0': bool, 'tied0':bool, 'diag1':bool, 'tied1':bool, 'minDCF':float})
df = df.astype({"components_0": int, 'components_1': int, 'diag0': bool, 'tied0':bool, 'diag1':bool, 'tied1':bool, 'minDCF':float})

for diag0, tied0, diag1, tied1 in product([True, False],[True, False],[True, False],[True, False]):
    maskp = (dfp.diag0 == diag0) & (dfp.diag1 == diag1) & (dfp.tied0 == tied0) & (dfp.tied1 == tied1)
    mask = (df.diag0 == diag0) & (df.diag1 == diag1) & (df.tied0 == tied0) & (df.tied1 == tied1)
    df_cur = df.loc[mask]
    dfp_cur = dfp.loc[maskp]
    plt.figure()
    for components in [2,4]:
        df_to_plot = df_cur[df_cur.components_1 == components].sort_values('components_0')
        dfp_to_plot = dfp_cur[dfp_cur.components_1 == components].sort_values('components_0')
        print(dfp_to_plot)
        plt.plot(dfp_to_plot.components_0, dfp_to_plot.minDCF, label=f'{components} target components, PCA 6')
        plt.plot(df_to_plot.components_0, df_to_plot.minDCF, label=f'{components} target components')
    if tied0 and diag0:
        non_target = 'TiedDiag'
    elif not tied0 and diag0:
        non_target = 'Diag'
    elif not tied0 and not diag0:
        non_target = 'Full'
    else:
        non_target = 'TiedFull'

    if tied1 and diag1:
        target = 'TiedDiag'
    elif not tied1 and diag1:
        target = 'Diag'
    elif not tied1 and not diag1:
        target = 'Full'
    else:
        target = 'TiedFull'

    plt.title(f'Non-target: {non_target} Target: {target}')
    plt.xlabel('Non-target components')
    plt.ylabel('minDCF')
    plt.legend()
    plt.savefig(f'./GMM_{non_target}_{target}.png', bbox_inches='tight')
