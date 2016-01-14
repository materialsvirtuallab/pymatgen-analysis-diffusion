# coding: utf-8

__author__ = "Iek-Heng Chu"
__version__ = 1.0
__date__ = "04/15"

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# to do list: add unittests, and ipython notebook examples

def gen_gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0)/2.0/np.power(sig,2.0))/np.sqrt(2.0*np.pi)/sig

class VanHoveAnalysis(object):

    """
        Class for van Hove function analysis. In particular, self-part (Gs) and
        distinct-part (Gd) of the van Hove correlation function G(r,t)
        for given species and given structure are computed. If you use this class,
        please consider citing the following paper:

        Zhu, Z.; Chu, I.-H.; Deng, Z. and Ong, S. P. "Role of Na+ Interstitials and Dopants
        in Enhancing the Na+ Conductivity of the Cubic Na3PS4 Superionic Conductor".
        Chem. Mater. (2015), 27, pp 8318–8325
    """


    def __init__(self, obj, avg_nsteps=50, Ntot=101, rmax=10.0, step_skip=50, sigma=0.1,
               species = ["Li","Na"]):
        """
        Initization.

        Args:
            obj (DiffusionAnalyzer): A DiffusionAnalyzer object
            avg_nsteps (int): Number of t0 used for statistical average
            Ntot (int): Number of radial grid points
            rmax (float): Maximum of radial grid (the minimum is always set zero)
            step_skip (int): # of time steps skipped during analysis. It defines
                        the resolution of the reduced time grid
            sigma (float): Smearing of a Gaussian function
            species ([string]): a list of specie symbols of interest
        """

        #initial check
        if step_skip <=0:
            raise ValueError("skip_step should be >=1!")

        nions, nsteps, ndim = obj.disp.shape

        if nsteps <= avg_nsteps:
            raise ValueError("Number of timesteps is too small!")

        ntsteps = nsteps - avg_nsteps

        if Ntot - 1 <= 0:
            raise ValueError("Ntot should be greater than 1!")

        if sigma <= 0.0:
            raise ValueError("sigma should be > 0!")

        dr = rmax / (Ntot-1)
        interval = np.linspace(0.0, rmax, Ntot)
        aux_factor = np.zeros_like(interval, dtype = np.double)
        gaussians = np.zeros((Ntot, Ntot), dtype = np.double)
        reduced_nt = int(ntsteps/float(step_skip)) + 1

        # reduced time grid
        rtgrid = np.arange(0.0, reduced_nt)
        # van Hove functions
        gsrt = np.zeros((reduced_nt, Ntot), dtype = np.double)
        gdrt = np.zeros((reduced_nt, Ntot), dtype = np.double)

        tracking_ions = []

        # auxiliary factor for 4*\pi*r^2
        for indx, v in enumerate(interval):
            if indx == 0:
                aux_factor[indx] = np.pi * dr ** 2
            else:
                aux_factor[indx] = 4.0 * np.pi * v ** 2

        for i, ss in enumerate(obj.get_drift_corrected_structures()):
            if i == 0:
                lattice = ss.lattice
                indices = [j for j, site in enumerate(ss) if site.specie.symbol in species]
                rho = float(len(indices)) / ss.lattice.volume

            all_fcoords = np.array(ss.frac_coords)
            tracking_ions.append(all_fcoords[indices,:])

        tracking_ions = np.array(tracking_ions)

        for indx, y in enumerate(interval):
            gaussians[indx,:] = gen_gaussian(interval, interval[indx], sigma) / float(avg_nsteps) / \
                            float(len(indices))

        # calculate self part of van Hove function
        image=np.array([0,0,0])
        for it in range(reduced_nt):
            dns = Counter()
            it0 = min(it*step_skip, ntsteps)
            for it1 in range(avg_nsteps):
                dists = [lattice.get_distance_and_image(tracking_ions[it1][u],
                        tracking_ions[it0+it1][u],
                        jimage=image)[0] for u in range(len(indices))]
                dists = filter(lambda e: e < rmax, dists)

                r_indices = [int(dist/dr) for dist in dists]
                dns.update(r_indices)

            for indx, dn in dns.most_common(Ntot):
                gsrt[it,:] += gaussians[indx, :] * dn

        # calculate distinct part of van Hove function of species
        r = np.arange(-1, 2)
        arange = r[:, None] * np.array([1, 0, 0])[None, :]
        brange = r[:, None] * np.array([0, 1, 0])[None, :]
        crange = r[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + brange[None, :, None] +crange[None, None, :]
        images = images.reshape((27, 3))

        # find the zero image vector
        zd = np.sum(images**2,axis=1)
        indx0 = np.argmin(zd)

        for it in range(reduced_nt):
            dns = Counter()
            it0 = min(it*step_skip, ntsteps)
            print it + 1, reduced_nt

            for it1 in range(avg_nsteps):
                dcf = tracking_ions[it0+it1,:,None,None,:] + images[None,None,:,:] \
                    - tracking_ions[it1,None,:,None,:]
                dcc = lattice.get_cartesian_coords(dcf)
                d2 = np.sum(dcc ** 2, axis=3)
                dists = [d2[u,v,j] ** 0.5 for u in range(len(indices)) for v in range(len(indices)) \
                    for j in range(27) if u != v or j != indx0]
                dists = filter(lambda e: e < rmax, dists)

                r_indices = [int(dist/dr) for dist in dists]
                dns.update(r_indices)

            for indx, dn in dns.most_common(Ntot):
                gdrt[it,:] += gaussians[indx,:] * dn / aux_factor[indx] / rho

        self.obj = obj
        self.avg_nsteps = avg_nsteps
        self.step_skip = step_skip
        self.rtgrid = rtgrid
        self.interval = interval
        self.gsrt = gsrt
        self.gdrt = gdrt

    def get_Gs_plots(self, figsize=(12,8)):
        """
        Plot self-part van Hove functions.
        """

        y = np.arange(np.shape(self.gsrt)[1]) * self.interval[-1] / float(len(self.interval)-1)
        timeskip = self.obj.time_step * self.obj.step_skip
        x = np.arange(np.shape(self.gsrt)[0]) * self.step_skip * timeskip / 1000.0
        X, Y = np.meshgrid(x,y, indexing="ij")

        ticksize = int(figsize[0] * 2.5)

        plt.figure(figsize=figsize, facecolor="w")
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        labelsize = int(figsize[0] * 3)

        plt.pcolor(X, Y, self.gsrt, cmap="jet", vmin=self.gsrt.min(), vmax=1.0)
        plt.xlabel("timesteps (ps)", size=labelsize)
        plt.ylabel("r ($\AA$)", size=labelsize)
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(ticks=[0,1]).set_label(label="4$\pi$G$_s$(t,r)", size=labelsize)
        plt.tight_layout()

        return plt

    def get_Gd_plots(self, figsize=(12,8)):
        """
        Plot distinct-part van Hove functions.
        """

        y = np.arange(np.shape(self.gdrt)[1]) * self.interval[-1] / float(len(self.interval)-1)
        timeskip = self.obj.time_step * self.obj.step_skip
        x = np.arange(np.shape(self.gdrt)[0]) * self.step_skip * timeskip / 1000.0
        X, Y = np.meshgrid(x,y, indexing="ij")

        ticksize = int(figsize[0] * 2.5)

        plt.figure(figsize=figsize, facecolor="w")
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        labelsize = int(figsize[0] * 3)

        plt.pcolor(X, Y, self.gdrt, cmap="jet", vmin=self.gdrt.min(), vmax=4.0)
        plt.xlabel("timesteps (ps)", size=labelsize)
        plt.ylabel("r ($\AA$)", size=labelsize)
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(ticks=[0,1,2,3,4]).set_label(label="G$_d$(t,r)", size=labelsize)
        plt.tight_layout()

        return plt