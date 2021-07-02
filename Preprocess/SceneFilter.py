# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import gc
from pyhdf.SD import SD, SDC
from skimage import io, color

import logging

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


class SceneFilter:
    """
    If these (separately) downloaded set of MODIS-derived variables exist:
     - NASA Worldview .jpeg images ('image')
     - Set of cloud products stored in .hdf files that contain
       ['Cloud_Mask_1km', 'Cloud_Top_Height', 'Cloud_Water_Path',
        'Cloud_Fraction', 'Cloud_Mask_1km']
    This class contains methods (in particular, filterScenes), to filter and
    process these in the following manner:
      1.  Read the data from the .hdf and .jpeg files and gather in a pandas
          DataFrame
      2.  Mask all pixels in all fields that have a zenith angle > zenmax
      3.  Subset the original fields to square areas of side npx that satisfy
          the max zenith angle criterion
      4.  Perturb each (disjoint) subset in 8 directions by dp, and measure
          if any resulting fields satisfy the following conditions:
              - Max zenith angle < zenmax
              - Overlap with any other perturbed scene < thrOv
              - Still completely inside the original field
      5.  Reject subset scene if > hcfr of the image is covered by clouds
          higher than hcThr
      6.  Find which other subsets overlap with each subset and store in
          separate .h5 file (ovl.h5) if saveOvl is True
      7.  Add the upper left corner's lat and lon to the subset's dataframe
      8.  Save each subset in a .h5
      9.  Plot if requested

      Input
      -----
      ppar  :  Dictionary containing the following parameters:
               sat        : Which satellite (Aqua or Terra)
               startDate  : First at which there is data to to filter
               endDate    : Last date at which there is data to to filter
               loadPath   : Path to load .jpeg and .hdf fields from
               savePath   : Path to store filtered .h5 fields in
               plot       : Boolean to plot each (large) scene and its accepted
                            subsets
               saveScenes : Boolean to store current filter sweep
               saveOvl    : Boolean to store overlap link information
               thrOv      : Minimum separation of overlapping scenes (pixels)
               dp         : How far to perturb original images (pixels)
               zenmax     : Maximum allowed zenith angle
               npx        : Pixels in a filtered scene
               thrCld     : Cloudy pixel classification threshold
               hcThr      : High cloud classification threshold
               hcfr       : Allowed high cloud fraction
               lat        : Range of latitudes of downloaded dataset
               lon        : Range on longitudes of downloaded dataset
    """

    def __init__(self, ppar):
        self.sat = ppar["sat"]
        self.startDate = ppar["startDate"]
        self.endDate = ppar["endDate"]
        self.loadPath = ppar["loadPath"]
        self.savePath = ppar["savePath"]
        self.plot = ppar["plot"]
        self.saveScenes = ppar["saveScenes"]
        self.saveOvl = ppar["saveOvl"]
        self.thrOv = ppar["thrOv"]
        self.dp = ppar["dp"]
        self.zenmax = ppar["zenmax"]
        self.npx = ppar["npx"]
        self.thrCl = ppar["thrCl"]
        self.hcThr = ppar["hcThr"]
        self.hcfr = ppar["hcfr"]
        self.lat = ppar["lat"]
        self.lon = ppar["lon"]

    def setSatPars(self, sat):
        """
        Set satellite-specific parameters.

        Parameters
        ----------
        sat : String
            Aqua or Terra.

        Returns
        -------
        selector : String
            Classifier for processing.
        svLab : String
            Extension to add to saved files.

        """

        if self.sat == "Aqua":
            selector = "MYD06"
            svLab = "a"
        elif self.sat == "Terra":
            selector = "MOD06"
            svLab = "t"
        return selector, svLab

    def getInfo(self, path, selector):
        """
        Get information on files to be processed.

        Parameters
        ----------
        path : String
            Absolute path to directory where all files to be processed are stored
        selector : String
            Aqua or Terra.

        Returns
        -------
        files : List
            List of .hdf files to be processed.
        imfiles : List
            List of .jpeg files to be processed.
        dates : List
            List of dates to be processed.
        fields : List
            List of scientific datasets (fields) to be processed.

        """
        files = []
        imfiles = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                fname = os.path.join(r, file)
                if selector in fname:
                    files.append(fname)
                if self.sat + "_CorrectedReflectance_TrueColor" in fname:
                    imfiles.append(fname)

        ## Check which cloud product fields we have on which dates
        fields = []
        dates = []
        for i in range(len(files)):
            spl = files[i].split(".")
            if not spl[-2] in fields:
                fields.append(spl[-2])
            if not spl[1] in dates:
                dates.append(spl[1])

        dates = np.asarray(dates)
        dates = np.sort(dates)

        return files, imfiles, dates, fields

    def d2yrd(self, date):
        """
        Convert date to yearday

        Parameters
        ----------
        date : Datetime date
            Date.

        Returns
        -------
        yrdays : String
            Yearday in correct format.

        """
        yrday = date.timetuple().tm_yday
        yrdays = str(yrday)
        if yrday < 100:
            if yrday < 10:
                yrdays = "00" + str(yrday)
            else:
                yrdays = "0" + str(yrday)
        return yrdays

    def applyZenithThreshold(self, df, date, zenmax):
        """
        Applies a zero mask to all pixels in all fields in df where the
        critical zenith angle exceeds zenmax

        Parameters
        ----------
        df : Pandas DataFrame
            DataFrame containing all fields as read by self.readData.
        date : Datetime date
            Date.
        zenmax : TYPE
            Critical zenith angle.

        Returns
        -------
        df : Pandas DataFrame
            Masked version of the imported DataFrame.
        rows : 1D Numpy array
            Rows where the mask has been applied.
        cols : 1D Numpy array
            Columns where the mask has been applied.

        """
        rows, cols = np.where(np.abs(df.loc[date, "Sensor_Zenith"]) > zenmax)
        for c in df.columns:
            try:
                df.loc[date, c][rows, cols] = 0
            except:
                print("Error setting zenith threshold in following field: ", c)
                continue
        return df, rows, cols

    def filterScene(self, img, npx, rows, cols):
        """
        Extract subsets of img of shape (npx,npx), nested in different
        locations within a critical zenith angle-masked field (img).
        Attempts are made in the following locations:
            - Bottom left corner
            - Top right corner
            - Top left corner
            - Bottom right corner
            - Internally in image:
                - From bottom left zero point
                - From top right zero point
        All successful extracts are returned.

        Parameters
        ----------
        img : 2D Numpy array with shape of downloaded scenes' extent
            Input scene of a field.
        npx : int
            Pixels in a filtered scene.
        rows : 1D Numpy array
            Rows extracted from self.applyZenithThreshold.
        cols : D Numpy array
            Columns extracted from self.applyZenithThreshold.

        Returns
        -------
        fpx : List
            List of (npx,npx) Numpy arrays, each containing a successfully
            extracted, disjoint scene.
        """

        fpx = []
        # Bottom left corner of image
        if img[img.shape[0] - 1 - npx : img.shape[0] - 1, :npx].all() != 0.0:
            p0x = 0
            p0y = img.shape[0] - 1 - npx
            fpx.append((p0y, p0x))
        # Top right corner of image
        if img[:npx, img.shape[1] - 1 - npx : img.shape[1] - 1].all() != 0.0:
            p0x = img.shape[1] - 1 - npx
            p0y = 0
            fpx.append((p0y, p0x))
        # Top left corner of image
        if img[:npx, :npx].all() != 0.0:
            p0x = 0
            p0y = 0
            fpx.append((p0y, p0x))
        # Bottom right corner of image
        if (
            img[
                img.shape[0] - 1 - npx : img.shape[0] - 1,
                img.shape[1] - 1 - npx : img.shape[1] - 1,
            ].all()
            != 0.0
        ):
            p0x = img.shape[1] - 1 - npx
            p0y = img.shape[0] - 1 - npx
            fpx.append((p0y, p0x))
        # Inside the image
        # Bottom left black point + (512x512)
        rbl = np.max(rows)
        cbl = cols[np.where(rows == rbl)[0][-1]] + 1
        if cbl + npx < img.shape[0]:
            if img[rbl - npx, cbl + npx] != 0.0:
                # Crop all fields to the specified area
                p0x = cbl
                p0y = rbl - npx
                fpx.append((p0y, p0x))
        # Top right black point - (512x512)
        rtr = np.min(rows)
        ctr = cols[np.where(rows == rtr)[0][0]] - 1
        if ctr - npx > 0 and rtr + npx < img.shape[0]:
            if img[rtr + npx, ctr - npx] != 0.0:
                p0x = ctr - npx
                p0y = rtr
                fpx.append((p0y, p0x))
        return fpx

    def perturb(self, img, fpx, npx, dp, thr):
        """
        1. Perturb all 'found' subscenes in 8 directions
        2. Reject all original or perturbed scenes that:
            - Fall outside the image
            - Are closer to any scene in accepted scenes from
             [original,perturbed] than a threshold value
        3. If not rejected, add scene to accepted (acce)

        Parameters
        ----------
        img : 2D Numpy array with shape of downloaded scenes' extent
            Input scene of a field.
        fpx : List
            List of (npx,npx) Numpy arrays, each containing a successfully
            extracted, disjoint subset.
        npx : int
            Number of pixels in a subset.
        dp : int
            Perturbation distance.
        thr : int
            Allowed overlap threshold.

        Returns
        -------
        acce : Numpy array of shape (nAccepted,npx,npx)
            Accepted, overlapping scenes, of which there are nAccepted.

        """

        if len(fpx) == 0:
            return []

        sh = img.shape
        orig = np.asarray(fpx)
        sten = np.array([-dp, 0, dp])
        grid = np.meshgrid(sten, sten)
        per0 = np.vstack((np.ravel(grid[0]), np.ravel(grid[1]))).T
        for i in range(len(orig)):
            oi = orig[i]
            pi = oi + per0

            # Filter perturbed images that would lie outside original image
            pi = pi[np.all(pi >= 0, axis=1)]  # Outside 0 edge
            pi = pi[np.all(pi + npx < sh[0], axis=1)]  # Outside shape[0] edge
            pi = pi[np.all(pi + npx < sh[1], axis=1)]  # Outside shape[1] edge

            # Filter perturbed images that would lie in a disallowed area
            check = np.empty(pi.shape[0], dtype="bool")
            for p in range(pi.shape[0]):
                check[p] = np.all(
                    img[pi[p][0] : pi[p][0] + npx, pi[p][1] : pi[p][1] + npx] != 0
                )
            pi = pi[check, :]

            # Filter images that lie too close to an accepted image
            if i == 0:
                acce = pi
            else:
                for j in range(len(pi)):
                    dist = np.linalg.norm(pi[j, :] - acce, axis=1)
                    if np.all(dist > thr):
                        acce = np.vstack((acce, pi[j, :]))
        return acce

    def bitsStripping(self, bit_start, bit_count, value):
        """
        Read binary fields - see e.g. https://science-emergence.com/Articles/How-to-read-a-MODIS-HDF-file-using-python-/

        Parameters
        ----------
        bit_start : int
            Start bit.
        bit_count : int
            Bit number.
        value : 2D Numpy array
            Binary input field.

        Returns
        -------
        2D Numpy array
            Float field.

        """

        bitmask = pow(2, bit_start + bit_count) - 1
        return np.right_shift(np.bitwise_and(value, bitmask), bit_start)

    def readData(self, date, files, imfiles, fields, thr, plot=False):
        """
        Parameters
        ----------
        date : Datetime date
            Date for which to read fields.
        files : List
            List of .hdf fields to load.
        imfiles : List
            List of .jpeg files to load.
        fields : List
            List of different fields to stored in the various files.
        thr : int
            Cloud certainty classification threshold.
            1 - certainly cloudy
            2 - likely cloudy
            3 - likely cloud free
            4 - certainly cloud free
        plot : Bool, optional
            Plot fields of read files. The default is False.

        Returns
        -------
        df : Pandas DataFrame
             2D fields ['image','Cloud_Water_Path','Cloud_Top_Height',
                        'Cloud_Fraction','Cloud_Mask_1km'] stored in columns.
        """

        # Initialise
        cols = fields.copy()
        cols.append("image")
        df = pd.DataFrame(index=[date], columns=cols)

        # Find and store the image
        datestrim = str(date.year) + date.strftime("%m") + date.strftime("%d")
        for s in range(len(imfiles)):
            if datestrim in imfiles[s]:
                img = io.imread(imfiles[s])
                img = color.rgb2gray(img)
                df.loc[date, "image"] = img

        # Find and store the fields
        datej = date.timetuple().tm_yday
        datejstr = str(datej)
        if datej < 100:
            if datej < 10:
                datejstr = "00" + str(datej)
            else:
                datejstr = "0" + str(datej)
        datestr = "A" + str(date.year) + datejstr
        for j in range(len(fields)):
            for s in range(len(files)):
                if all(x in files[s] for x in [datestr, fields[j]]):
                    try:
                        f = SD(files[s], SDC.READ)
                    except:
                        print("Unable to read", files[s])
                        continue
                    sds_obj = f.select(fields[j])  # select sds
                    if fields[j] == "Cloud_Mask_1km":
                        fie = self.bitsStripping(1, 2, sds_obj.get()[:, :, 0])
                        cl = np.where(fie < thr)
                        ncl = np.where(fie >= thr)
                        fie[cl[0], cl[1]] = 1
                        fie[ncl[0], ncl[1]] = 0
                    else:
                        offs = 0
                        sc = 1
                        fv = 0
                        for key, value in sds_obj.attributes().items():
                            if key == "add_offset":
                                offs = value
                            elif key == "scale_factor":
                                sc = value
                            elif key == "_FillValue":
                                fv = value
                        fie = sds_obj.get()
                        fie[fie == fv] = 0
                        fie = (fie - offs) * sc
                    if fields[j] == "Cloud_Mask_1km":
                        cl = np.where(fie >= 0.5)
                        ncl = np.where(fie < 0.5)
                        fie[cl[0], cl[1]] = 1
                        fie[ncl[0], ncl[1]] = 0
                    df.loc[date, fields[j]] = fie

        if plot:
            fig, axs = plt.subplots(ncols=6, figsize=(10, 5))
            axs[0].imshow(df["image"].values[0], "gray")
            axs[0].set_title("image")
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[1].imshow(df["Cloud_Mask_1km"].values[0], "gray")
            axs[1].set_title("cloud mask")
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            axs[2].imshow(df["Cloud_Fraction"].values[0], "gray")
            axs[2].set_title("cloud fraction")
            axs[2].set_xticks([])
            axs[2].set_yticks([])
            axs[3].imshow(df["Cloud_Top_Height"].values[0], "gray")
            axs[3].set_title("cloud top height")
            axs[3].set_xticks([])
            axs[3].set_yticks([])
            axs[4].imshow(df["Cloud_Water_Path"].values[0], "gray", vmax=250)
            axs[4].set_title("cwp")
            axs[4].set_xticks([])
            axs[4].set_yticks([])
            sc = axs[5].imshow(df["Sensor_Zenith"].values[0], "gray")
            axs[5].set_title("zenith angle")
            axs[5].set_xticks([])
            axs[5].set_yticks([])
            cbax = fig.add_axes([0.92, 0.38, 0.025, 0.24])
            fig.colorbar(sc, cax=cbax)
            # plt.savefig('uncutFields.png',dpi=300,bbox_inches='tight')
            plt.show()

        return df

    def filterScenes(self):
        """
        Main filtering function. See class description for details.

        """

        selector, svLab = self.setSatPars(self.sat)

        files, imfiles, dates, fields = self.getInfo(self.loadPath, selector)

        ## Filter ''good'' scenes from the dates/scenes

        d0 = np.where(
            dates == "A" + str(self.startDate.year) + self.d2yrd(self.startDate)
        )[0][0]
        de = np.where(dates == "A" + str(self.endDate.year) + self.d2yrd(self.endDate))[
            0
        ][0]

        if self.saveOvl:
            dfOvl = pd.DataFrame(columns=["dist", "iovl"])

        for i in range(d0, de + 1):

            # Set the date
            date = dates[i][1:]
            year = int(date[:4])
            day = int(date[4:])
            date = datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1)
            print(date.date())

            # Clean the previously loaded dataframe from memory
            if "df" in locals():
                del df
                gc.collect()

            # Read a new dataframe
            df = self.readData(date, files, imfiles, fields, self.thrCl, self.plot)

            # Set excess zenith angle
            df, rows, cols = self.applyZenithThreshold(df, date, self.zenmax)

            # FILTERING
            # Try to find a field of size npx x npx that satisfies the criteria
            # - Finds multiple of these
            # - Perturbs found images in a certain direction by npxp and
            #   checks if that image is still valid
            img = df.loc[date, "image"].copy()
            fpx = self.filterScene(img, self.npx, rows, cols)
            acc = self.perturb(img, fpx, self.npx, self.dp, self.thrOv)

            print("Number of subscenes:", len(acc))
            for j in range(len(acc)):
                # Create new df
                ind = str(date.date()) + "-" + str(svLab) + "-" + str(j)
                dfj = pd.DataFrame(index=[ind], columns=df.columns)
                rejected = False

                # Populate
                for c in dfj.columns:
                    dfj.loc[ind, c] = df.loc[date, c][
                        acc[j][0] : acc[j][0] + self.npx,
                        acc[j][1] : acc[j][1] + self.npx,
                    ]

                # Reject if due to something weird there are still black px
                if not rejected:
                    if dfj.loc[ind, "image"].any() == 0.0:
                        rejected = True
                        print("Still reject")

                # If there are too many high clouds, reject
                if not rejected:
                    cth = dfj["Cloud_Top_Height"].values[0].copy()
                    if len(np.where(cth > self.hcThr)[0]) / cth.size > self.hcfr:
                        rejected = True
                        print(j, ": Reject (high clouds)")

                # PROCESS FILTERED SCENES
                if not rejected:
                    # Set negative values to 0
                    for c in dfj.columns:
                        if len(dfj.loc[ind, c].shape) == 2:
                            r0, c0 = np.where(dfj.loc[ind, c] < 0)
                            dfj.loc[ind, c][r0, c0] = 0

                # Find scenes in the original image that this scene overlaps with
                # Can still identify a rejected scene as one overlapping with the anchor
                if not rejected and self.saveOvl:
                    dfind = str(date.date()) + "-" + svLab + "-" + str(j)
                    dfOvl = dfOvl.reindex(dfOvl.index.tolist() + [dfind])
                    if len(acc) > 1:
                        dists = np.linalg.norm(acc - acc[j, :], axis=1)
                        iovl = np.argpartition(dists, 1)[1]
                        ovl = dists[iovl]
                        dfOvl["dist"].loc[dfind] = ovl
                        dfOvl["iovl"].loc[dfind] = (
                            str(date.date()) + "-" + svLab + "-" + str(iovl)
                        )

                # Find the (lat,lon) of the top left pixel and add to the df
                lon0 = (
                    self.lon[0] + acc[j][1] * (self.lon[1] - self.lon[0]) / img.shape[1]
                )
                lat0 = (
                    self.lat[1] + acc[j][0] * (self.lat[0] - self.lat[1]) / img.shape[0]
                )
                dfj["lon"] = lon0
                dfj["lat"] = lat0

                # SAVE TO FOLDER
                if not rejected and self.saveScenes:
                    dfj.to_hdf(
                        self.savePath
                        + "/"
                        + str(date.date())
                        + "-"
                        + svLab
                        + "-"
                        + str(j)
                        + ".h5",
                        "sds_filtered",
                        mode="w",
                    )

                if self.plot:
                    fig, axs = plt.subplots(ncols=6, figsize=(10, 5))
                    axs[0].imshow(dfj["image"].values[0], "gray")
                    axs[0].set_title("image")
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])
                    axs[1].imshow(dfj["Cloud_Mask_1km"].values[0], "gray")
                    axs[1].set_title("cloud mask")
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    axs[2].imshow(dfj["Cloud_Fraction"].values[0], "gray")
                    axs[2].set_title("cloud fraction")
                    axs[2].set_xticks([])
                    axs[2].set_yticks([])
                    axs[3].imshow(dfj["Cloud_Top_Height"].values[0], "gray")
                    axs[3].set_title("cloud top height")
                    axs[3].set_xticks([])
                    axs[3].set_yticks([])
                    axs[4].imshow(dfj["Cloud_Water_Path"].values[0], "gray", vmax=250)
                    axs[4].set_title("cwp")
                    axs[4].set_xticks([])
                    axs[4].set_yticks([])
                    axs[5].imshow(dfj["Sensor_Zenith"].values[0], "gray")
                    axs[5].set_title("zenith angle")
                    axs[5].set_xticks([])
                    axs[5].set_yticks([])
                    if rejected:
                        axs[0].annotate("REJECTED", (0, 510), fontsize=15, zorder=10)
                    plt.show()

        if self.saveOvl:
            dfOvl.to_hdf(
                self.savePath + "/../ovl-" + svLab + ".h5", "overlap", mode="w"
            )


if __name__ == "__main__":
    ppar = {
        "sat": "Aqua",
        "startDate": datetime.datetime(2002, 12, 1),
        "endDate": datetime.datetime(2002, 12, 2),
        "loadPath": "/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/testEnv/Data/Download/DataAqua",
        "savePath": "/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/testEnv/Data/Filtered",
        "plot": True,
        "saveScenes": True,
        "saveOvl": True,
        "thrOv": 200,  # Minimum separation of overlapping scenes (pixels)
        "dp": 256,  # How far to perturb original images?
        "zenmax": 45,  # Max allowed zenith angle
        "npx": 512,  # Pixels in a filtered scene
        "thrCl": 1,  # Cloudy pixel classification threshold
        "hcThr": 5000,  # High cloud classification threshold
        "hcfr": 0.2,  # Allowed high cloud fraction
        "lat": [10, 20],
        "lon": [-58, -48],
    }
    sceneFilter = SceneFilter(ppar)
    sceneFilter.filterScenes()
