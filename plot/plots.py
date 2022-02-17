from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
def embedding_plot(rep:"ndarray(number,dim)",label:"ndarray(number)"):
    print('ploting fig----------------------')
    X_embedded = TSNE(n_components=2,init='random').fit_transform(rep.reshape(rep.shape[0],-1))
    
    N=10
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    scat = ax.scatter(X_embedded[:,0],X_embedded[:,1],c=label,cmap=cmap,     norm=norm)
    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cb.set_label('Custom cbar')
    