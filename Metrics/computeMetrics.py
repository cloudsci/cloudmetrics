#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

from .createDataFrame import getAllMetrics
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
from .utils import anyInList


def computeMetrics(metrics, mpar):
    """
    Compute all metrics in given list, with a set of input parameters mpar, by
    sequentially calling the compute method of all desired metrics. The metric
    classes (specifially their compute methods) require mpar to define several
    parameters, which must therefore be set up before this function can be
    called.

    Parameters
    ----------
    metrics : list
        List of all metrics to be computed.
    mpar : Dict
    Specifies the following parameters:
        loadPath : Path to load .h5 files that contain a pandas dataframe
                   with a cloud mask field as one of the columns.
        savePath : Path to a .h5 containing a pandas dataframe whose columns
                   contain metrics and whose indices are scenes. Two of
                   these columns can be filled by 'scai' and 'd0'.
        save     : Boolean to specify whether to store the variables in
                   savePath/Metrics.h5
        resFac   : Resolution factor (e.g. 0.5), to coarse-grain the field.
        plot     : Boolean to specify whether to make plot with details on
                   this metric for each scene.
        con      : Connectivitiy for segmentation (1 - 4 seg, 2 - 8 seg)
        areaMin  : Minimum cloud size considered in computing metric
        fMin     : First scene to load
        fMax     : Last scene to load. If None, is last scene in set.
        fields   : Naming convention for fields, used to set the internal
                   field to be used to compute each metric.

    Returns
    -------
    None.

    """

    if "cf" in metrics:
        print("Computing cf")
        cf = CF(mpar=mpar)
        cf.compute()

    cwpList = ["cwp", "cwpVar", "cwpVarCl", "cwpSke", "cwpKur"]
    if anyInList(metrics, cwpList):
        print("Computing cwp metrics")
        cwp = CWP(mpar=mpar)
        cwp.compute()

    objectList = ["lMax", "lMean", "nClouds", "eccA", "periSum"]
    if anyInList(metrics, objectList):
        print("Computing object metrics")
        objects = Objects(mpar=mpar)
        objects.compute()

    cthList = ["cth", "cthVar", "cthSke", "cthKur"]
    if anyInList(metrics, cthList):
        print("Computing cth metrics")
        cth = CTH(mpar=mpar)
        cth.compute()

    if "sizeExp" in metrics:
        print("Computing sizeExp")
        csd = CSD(mpar=mpar)
        csd.compute()

    fourList = ["beta", "betaa", "specL", "specLMom", "psdAzVar"]
    if anyInList(metrics, fourList):
        print("Computing Fourier metrics")
        fourier = FourierMetrics(mpar=mpar)
        fourier.compute()

    if "cop" in metrics:
        print("Computing COP")
        cop = COP(mpar=mpar)
        cop.compute()

    scaiList = ["scai", "d0"]
    if anyInList(metrics, scaiList):
        print("Computing SCAI")
        scai = SCAI(mpar=mpar)
        scai.compute()

    rdfList = ["rdfMax", "rdfInt", "rdfDiff"]
    if anyInList(metrics, rdfList):
        print("Computing RDF metrics")
        rdf = RDF(mpar=mpar)
        rdf.compute()

    networkList = [
        "netVarDeg",
        "netAWPar",
        "netCoPar",
        "netLPar",
        "netLCorr",
        "netDefSl",
        "netDegMax",
    ]
    if anyInList(metrics, networkList):
        print("Computing network metrics")
        network = Network(mpar=mpar)
        network.compute()

    if "iOrgPoiss" in metrics:
        print("Computing Poisson iOrg")
        iOrgPoisson = IOrgPoisson(mpar=mpar)
        iOrgPoisson.compute()

    if "fracDim" in metrics:
        print("Computing fractal dimension")
        fracDim = FracDim(mpar=mpar)
        fracDim.compute()

    if "iOrg" in metrics:
        print("Computing iOrg")
        iOrg = IOrg(mpar=mpar)
        iOrg.compute()

    osList = ["os", "osAv"]
    if anyInList(metrics, osList):
        print("Computing open sky metric")
        os = OpenSky(mpar=mpar)
        os.compute()

    if "twpVar" in metrics:
        twpVar = TWPVar(mpar=mpar)
        twpVar.compute()

    woiList = ["woi1", "woi2", "woi3", "woi"]
    if anyInList(metrics, woiList):
        print("Computing wavelet organisation indicies")
        woi = WOI(mpar=mpar)
        woi.compute()

    if "orie" in metrics:
        print("Computing orientation from raw image moments")
        orie = Orient(mpar=mpar)
        orie.compute()


def evaluateMetrics(metrics, fields, mpar=None):
    """
    Compute metrics on a set of input fields on a single scene, by sequentially
    calling the metric method of metric objects. This may be more flexible
    than using compute methods if one only wishes to compute one metric and
    have it in memory immediately. This method can also be called without
    passing mpar to the metric object upon instantiation. However, this will
    set the parameters plot, con and areaMin to their defaults (False, 1, 4).
    For all metrics to be computable, one must at least have cloud mask, image,
    cloud water path cloud-top height available. See main.py for an example of
    how to use this function.

    Parameters
    ----------
    metrics : list
        List of metrics to be computed.
    fields : dict
        Dictionary of 2D numpy arrays. If one wishes to compute all metrics,
        all the following dictionary entires must be specified:
            - 'cm'  (Cloud mask)
            - 'im'  (Image)
            - 'cwp' (Cloud water path)
            - 'cth' (Cloud-top height)
    mpar : dict, optional
        General metric parameters (same as for computeMetrics). The default is
        None.

    Returns
    -------
    df : Pandas dataframe
        Dataframe of a single row containing metrics for the provided.

    """

    metrics = getAllMetrics(metrics)
    df = pd.DataFrame(index=[0], columns=metrics)

    if "cf" in metrics:
        print("Computing cf")
        cf = CF(mpar=mpar)
        df["cf"] = cf.metric(fields["cm"])
    if "cwp" in metrics:
        print("Computing cwp metrics")
        cwp = CWP(mpar=mpar)
        (
            df["cwp"],
            df["cwpVar"],
            df["cwpSke"],
            df["cwpKur"],
            df["cwpVarCl"],
        ) = cwp.metric(fields["cwp"])
    if "lMax" in metrics:
        print("Computing object metrics")
        objects = Objects(mpar=mpar)
        (
            df["lMax"],
            df["lMean"],
            df["nClouds"],
            df["eccA"],
            df["periSum"],
        ) = objects.metric(fields["cm"], fields["im"])
    if "cth" in metrics:
        print("Computing cth metrics")
        cth = CTH(mpar=mpar)
        df["cth"], df["cthVar"], df["cthSke"], df["cthKur"] = cth.metric(
            fields["cth"], fields["cm"]
        )
    if "sizeExp" in metrics:
        print("Computing sizeExp")
        csd = CSD(mpar=mpar)
        df["sizeExp"] = csd.metric(fields["cm"])
    if "beta" in metrics:
        print("Computing Fourier metrics")
        fourier = FourierMetrics(mpar=mpar)
        (
            df["beta"],
            df["betaa"],
            df["psdAzVar"],
            df["specL"],
            df["specLMom"],
        ) = fourier.metric(fields["cm"])
    if "cop" in metrics:
        print("Computing COP")
        cop = COP(mpar=mpar)
        df["cop"] = cop.metric(fields["cm"])
    if "scai" in metrics:
        print("Computing SCAI")
        scai = SCAI(mpar=mpar)
        df["d0"], df["scai"] = scai.metric(fields["cm"])
    if "rdfMax" in metrics:
        print("Computing RDF metrics")
        rdf = RDF(mpar=mpar)
        df["rdfMax"], df["rdfInt"], df["rdfDiff"] = rdf.metric(fields["cm"])
    if "netVarDeg" in metrics:
        print("Computing network metrics")
        network = Network(mpar=mpar)
        (
            df["netVarDeg"],
            df["netAWPar"],
            df["netCoPar"],
            df["netLPar"],
            df["netLCorr"],
            df["netDefSl"],
            df["netDegMax"],
        ) = network.metric(fields["cm"])
    if "iOrgPoiss" in metrics:
        print("Computing Poisson iOrg")
        iOrgPoisson = IOrgPoisson(mpar=mpar)
        df["iOrgPoiss"] = iOrgPoisson.metric(fields["cm"])
    if "fracDim" in metrics:
        print("Computing fractal dimension")
        fracDim = FracDim(mpar=mpar)
        df["fracDim"] = fracDim.metric(fields["cm"])
    if "iOrg" in metrics:
        print("Computing iOrg")
        iOrg = IOrg(mpar=mpar)
        df["iOrg"] = iOrg.metric(fields["cm"])
    if "os" in metrics:
        print("Computing open sky metric")
        os = OpenSky(mpar=mpar)
        df["os"] = os.metric(fields["cm"])
    if "twpVar" in metrics:
        twpVar = TWPVar(mpar=mpar)
        df["twpVar"] = twpVar.metric(fields["cwp"])
    if "woi3" in metrics:
        print("Computing wavelet organisation indicies")
        woi = WOI(mpar=mpar)
        df["woi1"], df["woi2"], df["woi3"], df["woi"] = woi.metric(
            fields["cwp"], fields["cm"]
        )
    if "orie" in metrics:
        print("Computing orientation from raw image moments")
        orie = Orient(mpar=mpar)
        df["orie"] = orie.metric(fields["cm"])

    return df
