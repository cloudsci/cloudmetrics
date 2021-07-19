import urllib.request
import os
import pandas as pd


def downloadMODISImgs(
    startDate,
    endDate,
    extent,
    savePath,
    satellite="Aqua",
    lon2pix=111.4,
    lat2pix=111,
    exist_skip=False,
    var="CorrectedReflectance_TrueColor",
    filetype="jpeg",
):

    """
    Download images from NASA Worldview (https://worldview.earthdata.nasa.gov/)
    using direct urllib retrieval.

    Input
    -----
    startDate  : First date to download an image from
    endDate    : Last date to download an image from
    extent     : Longitude and latitude range to subset.
                 Use convention [lonMin, lonMax, latMin, latMax]
    savePath   : Path where to store downloaded images
    satellite  : Which satellite to retrieve images from.
                 Options are {'Aqua', 'Terra'}.
    lon2pix    : Conversion factor from a degree of longitude to a pixel
    lat2pix    : Conversion factor from a degree of latitude to a pixel
    exist_skip : Skip images that already exist in savePath
    var        : Worldview layer to download.
                 Defaults to True Corrected Reflectance.
    filetype   : File type to store image in. Defaults to .jpeg

    """

    lon1 = extent[0]
    lon2 = extent[1]
    lat1 = extent[2]
    lat2 = extent[3]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    loc = f"&BBOX={lat1},{lon1},{lat2},{lon2}"
    loc_str = f"_{lon1}-{lon2}_{lat1}-{lat2}"
    size = f"&WIDTH={int(dlon * lon2pix)}&HEIGHT={int(dlat * lat2pix)}"
    layer = f"&LAYERS=MODIS_{satellite}_{var},Coastlines"

    dateRange = pd.date_range(start=startDate, end=endDate)

    for i in range(len(dateRange)):
        date = dateRange[i].date()
        print(date)

        yr = str(date.year)
        mon = date.strftime("%m")
        day = date.strftime("%d")

        url = (
            "https://wvs.earthdata.nasa.gov/api/v1/snapshot?"
            + "REQUEST=GetSnapshot&TIME="
            + yr
            + "-"
            + mon
            + "-"
            + day
            + loc
            + "&CRS=EPSG:4326"
            + layer
            + "&FORMAT=image/"
            + filetype
            + size
        )
        save_str = (
            savePath
            + f"/{satellite}_"
            + var
            + yr
            + date.strftime("%m")
            + "{:02d}".format(date.day)
            + loc_str
            + "."
            + filetype
        )
        if exist_skip and os.path.exists(save_str):
            print("Skip")
        else:
            try:
                urllib.request.urlretrieve(url, save_str)
            except:
                print(f"Download failed for {save_str}")


if __name__ == "__main__":
    downloadMODISImgs("2002-12-01", "2002-12-02", [-58, -48, 10, 20], ".")
