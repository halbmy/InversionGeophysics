import numpy as np
from matplotlib import pyplot as plt
import pygimli as pg
from pygimli.physics.ves import VESModelling
# from pygimli.physics.em import FDEM
# from pygimli.physics.em.fdem import HEM1dWithElevation


def showPhiD(x, y, mat, xlabel=r"$\rho_2$ ($\Omega$m)", ylabel=r"$d_2$ (m)",
            vmin=1, vmax=4, orientation="vertical"):
    fig, ax = plt.subplots()
    im = ax.matshow(mat, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    # im = ax.matshow(np.log10(mat), cmap="Spectral_r", vmin=1.5, vmax=4.5)
    xt = np.arange(0, len(x), 5)
    xtl = ["{:.0f}".format(np.round(xx)) for xx in x[xt]]
    yt = np.arange(0, len(y), 5)
    ytl = ["{:.0f}".format(np.round(yy)) for yy in y[yt]]
    plt.xticks(xt, xtl)
    plt.yticks(yt, ytl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticks_position("bottom")
    cb = plt.colorbar(im, orientation=orientation)

    return ax, cb

if __name__ == "__main__":
    ab2 = np.logspace(0.5, 3, 50)
    ves = VESModelling(ab2=ab2, mn2=ab2/3)
    # %%
    model = [20, 20, 100, 1000, 100]
    resp = ves.response(model)
    thk2 = np.logspace(0.5, 2.5, 31)
    res2 = np.logspace(2, 4, 41)
    # %%
    model1 = np.array(model)
    M1 = np.zeros((len(thk2), len(res2)))
    for i, t in enumerate(thk2):
        model1[1] = t
        for j, r in enumerate(res2):
            model1[3] = r
            resp1 = ves.response(model1)
            M1[i, j] = np.sum((np.log(resp1)-np.log(resp))**2/0.1**2)

    showPhiD(res2, thk2, np.log10(M1))
    # sdffs
    # pg.plt.matshow(np.log(M1))
    # %%
    f = np.logspace(2, 5, 10)
    fop = pg.core.FDEM1dModelling(3, f, 10., 50)
    resp = fop.response(model)
    # %%
    model1 = np.array(model)
    M2 = np.zeros((len(thk2), len(res2)))
    for i, t in enumerate(thk2):
        model1[1] = t
        for j, r in enumerate(res2):
            model1[3] = r
            resp1 = fop.response(model1)
            M2[i, j] = np.sum((resp1-resp)**2/0.2**2)

    showPhiD(res2, thk2, np.log10(M2))
    # pg.plt.matshow(np.log(M2))
    # %%
    showPhiD(res2, thk2, np.log10((M1+M2)/2))
    # pg.plt.matshow(np.log(M1+M2*40))
    # %%
    fig, ax = plt.subplots(figsize=(3, 3))
    pg.viewer.mpl.drawModel1D(ax, [20, 20, 20], [100, 1000, 100, 100])#,
    ax.set_xlim(0, 1050)
    ax.text(100, 10, r" $d_1$")
    ax.text(1000, 30, r" $d_2$", ha="right")
    ax.text(100, 22, r"$\rho_1$", ha="center", va="top")
    ax.text(1000, 42, r"$\rho_2$", ha="center", va="top")
    ax.text(100, 38, r"$\rho_3$", ha="center")
