import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline










def read_file(f, maxx=-1):
    res = [[],[],[],[],[],[],[],[],[]]
    for line in f:
        idx = 0
        line = line.replace('\n','').split(' ')
        if 'valid' in line[0]:
            idx = 1
            epoch = int(line[0].split(':')[0].replace('epoch','').replace('train','').replace('valid',''))
            res[-1].append(epoch)
        line = line[1:]
        res[idx*4+0].append(float(line[0]))
        res[idx*4+1].append(float(line[1]))
        tloss = float(line[0]) + float(line[1])
        res[idx*4+2].append(tloss)
        res[idx*4+3].append(min(tloss, res[idx*4+3][-1] if len(res[idx*4+3])>0 else 1e20))
    for i in range(len(res)):
        res[i] = res[i][:maxx]
    return res


def plot_smooth(x,y,sm=500,title=''):
    X_ = np.array(x)
    Y_ = np.array(y)
    # X_Y_Spline = make_interp_spline(X_, Y_)
    # X_ = np.linspace(X_.min(), X_.max(), sm)
    # Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, label=title)



maps = {}
f = open('../logs/train_log_color.log','r')
maps['color'] = read_file(f)
f.close()

f = open('../logs/train_log_uniform.log','r')
maps['uniform'] = read_file(f)
f.close()

f = open('../logs/train_log_vanilla.log','r')
maps['vanilla'] = read_file(f)
f.close()

plot_smooth(maps['color'][-1],maps['color'][7],sm=500,title='Coloring')
plot_smooth(maps['uniform'][-1],maps['uniform'][7],sm=500,title='uniform')
plot_smooth(maps['vanilla'][-1],maps['vanilla'][7],sm=500,title='vanilla')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.savefig('../figs/valid.png')

plt.clf()
plot_smooth(maps['color'][-1],maps['color'][3],sm=500,title='Coloring')
plot_smooth(maps['uniform'][-1],maps['uniform'][3],sm=500,title='uniform')
plot_smooth(maps['vanilla'][-1],maps['vanilla'][3],sm=500,title='vanilla')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.savefig('../figs/train.png')


for k in maps:
    print(f'{k} best train loss:  {maps[k][3][-1]}')
for k in maps:
    print(f'{k} best val loss:  {maps[k][7][-1]}')
