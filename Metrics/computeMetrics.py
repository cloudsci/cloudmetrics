#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .cf import CF
from .cwp import CWP
from .objects import Objects
from .cth import CTH
from .csd import CSD
from .fourier import FourierMetrics
from .cop import COP
from .scai import SCAI
from .rdf import RDF
from .network import Network
from .iorgPoisson import IOrgPoisson
from .fracDim import FracDim
from .iorg import IOrg
from .openSky import OpenSky
from .twpVar import TWPVar
from .woi import WOI
from .orientation import Orient

def computeMetrics(metrics,mpar):
    if 'cf' in metrics:
        print('Computing cf')
        cf = CF(mpar)
        cf.compute()
    if 'cwp' in metrics:
        print('Computing cwp metrics')
        cwp = CWP(mpar)
        cwp.compute()
    if 'lMax' in metrics:
        print('Computing object metrics')
        objects = Objects(mpar)
        objects.compute()
    if 'cth' in metrics:
        print('Computing cth metrics')
        cth = CTH(mpar)
        cth.compute()
    if 'sizeExp' in metrics:
        print('Computing sizeExp')
        csd = CSD(mpar)
        csd.compute()
    if 'beta' in metrics:
        print('Computing Fourier metrics')
        fourier = FourierMetrics(mpar)
        fourier.compute()
    if 'cop' in metrics:
        print('Computing COP')
        cop = COP(mpar)
        cop.compute()
    if 'scai' in metrics:
        print('Computing SCAI')
        scai = SCAI(mpar)
        scai.compute()
    if 'rdfMax' in metrics:
        print('Computing RDF metrics')
        rdf = RDF(mpar)
        rdf.compute()
    if 'netVarDeg' in metrics:
        print('Computing network metrics')
        network = Network(mpar)
        network.compute()
    if 'iOrgPoiss' in metrics:
        print('Computing Poisson iOrg')
        iOrgPoisson = IOrgPoisson(mpar)
        iOrgPoisson.compute()
    if 'fracDim' in metrics:
        print('Computing fractal dimension') 
        fracDim = FracDim(mpar)
        fracDim.compute()
    if 'iOrg' in metrics:
        print('Computing iOrg')
        iOrg = IOrg(mpar)
        iOrg.compute()
    if 'os' in metrics:
        print('Computing open sky metric')
        os = OpenSky(mpar)
        os.compute()
    if 'twpVar' in metrics:
        twpVar = TWPVar(mpar)
        twpVar.compute()
    if 'woi3' in metrics:
        print('Computing wavelet organisation indicies')
        woi = WOI(mpar)
        woi.compute()
    if 'orie' in metrics:
        print('Computing orientation from raw image moments')
        orie = Orient(mpar)
        orie.compute()
    