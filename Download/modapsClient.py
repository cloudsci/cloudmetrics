#!/usr/bin/env python
# encoding: utf-8
"""
**Module modapsclient**

**A client to do talk to MODAPS web services.
See http://ladsweb.nascom.nasa.gov/data/web_services.html**

*Created by Chris Waigl on 2018-03-25, by moving module out of pygaarst.*
*Extended and included to directly download and postprocess on 2020-02-20.*
"""

from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals
from future import standard_library

standard_library.install_aliases()

from builtins import filter
from builtins import object
import sys
import urllib.request, urllib.error, urllib.parse
import logging
from xml.dom import minidom

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("pygaarst.modapsclient")

MODAPSBASEURL = (
    "https://modwebsrv.modaps.eosdis.nasa.gov/" "axis2/services/MODAPSservices"
)
MODAPSBASEURL_noTLS = (
    "http://modwebsrv.modaps.eosdis.nasa.gov/" "axis2/services/MODAPSservices"
)


def _parsekeyvals(domobj, containerstr, keystr, valstr):
    """Parses what's basically a keys-values structrue in a list of container
    elements of the form
    <container><key>A_KEY</key><value>A_VALUE</value></container>
    and returns it as a dictionary"""
    output = {}
    nodelist = domobj.getElementsByTagName(containerstr)
    for node in nodelist:
        key = value = None
        children = node.childNodes
        for child in children:
            if child.tagName == keystr:
                key = child.firstChild.data
            elif child.tagName == valstr:
                value = child.firstChild.data
        if key and value:
            output[key] = value
    return output


def _parselist(domobj, containerstr):
    """Parses what's basically a list of strings contained in a container
    elements of the form
    <container>A_VALUE</container>
    and returns it as a list"""
    output = []
    nodelist = domobj.getElementsByTagName(containerstr)
    for node in nodelist:
        output.append(node.firstChild.data)
    return output


def _parselistofdicts(domobj, containerstr, prefix, listofkeys):
    """Parses what's basically a list of dictionaries contained in a container
    elements of the form
    <container><prefix:key1>VAL1</prefix:key1><prefix:key2>...
    </prefix:key2>...</container>
    and returns it as a list of dictionaries"""
    output = []
    nodelist = domobj.getElementsByTagName(containerstr)
    for node in nodelist:
        item = {}
        children = node.childNodes
        for child in children:
            if child.tagName.startswith(prefix):
                key = child.tagName.replace(prefix, "", 1)
                item[key] = child.firstChild.data
        output.append(item)
    return output


def _startswithax(item):
    """
    Helper function to filter tag lists for starting with xmlns:ax
    """
    return item.startswith("xmlns:ax")


class ModapsClient(object):
    """
    Implements a client for MODAPS web service retrieval of satellite data,
    without post-processing

    See http://ladsweb.nascom.nasa.gov/data/quickstart.html
    """

    def __init__(self):
        self.baseurl = MODAPSBASEURL
        self.headers = {"User-Agent": "satellite RS data fetcher"}

    def _rawresponse(self, url, data=None):
        if data:
            for tag in data:
                if type(data[tag]) == list:
                    data[tag] = ",".join(data[tag])
            querydata = urllib.parse.urlencode(data).encode("utf-8")
            request = urllib.request.Request(url, querydata, headers=self.headers)
        else:
            request = urllib.request.Request(url, headers=self.headers)
        try:
            response = urllib.request.urlopen(request).read()
        except urllib.error.HTTPError as err:
            logging.critical("Error opening URL: %s" % err)
            logging.critical("URL is %s" % url)
            if data:
                logging.critical("Query string is %s" % querydata)
            raise
        return response

    def _makeurl(self, path, TLS=True):
        if TLS:
            return self.baseurl + path
        return MODAPSBASEURL_noTLS + path

    def _parsedresponse(self, path, argdict, parserfun, data=None, unstabletags=False):
        """Returns response based on request and parser function"""
        url = self._makeurl(path, TLS=True)
        try:
            response = self._rawresponse(url, data=data)
        except urllib.error.HTTPError:
            try:
                url = self._makeurl(path, TLS=False)
                response = self._rawresponse(url, data=data)
            except urllib.error.HTTPError:
                logging.critical("Tried with and without TLS. Web service unavailable.")
                raise
        xmldoc = minidom.parseString(response)
        if unstabletags:
            attr = list(xmldoc.documentElement.attributes.items())
            pref = list(filter(_startswithax, [item[0] for item in attr]))
            if not pref:
                LOGGER.error("No valid namespace prefix found for request %s ." % url)
                sys.exit(1)
            elif len(pref) > 1:
                LOGGER.error(
                    "Too many potential namespace prefix found for "
                    + "request %s. Using %s." % (url, pref[0])
                )
            else:
                prefix = pref[0][6:] + ":"
                for item in argdict:
                    if item != "containerstr":
                        argdict[item] = prefix + argdict[item]
        return parserfun(xmldoc, **argdict)

    def getAllOrders(self, email):
        """All orders for an email address"""
        raise NotImplementedError(
            "Method {} not implemented. Probably won't be.".format("getAllOrders")
        )

    def getBands(self, product):
        """Available bands for a product"""
        path = "/getBands"
        parser = _parsekeyvals
        argdict = {}
        argdict["containerstr"] = "return"
        argdict["keystr"] = "mws:name"
        argdict["valstr"] = "mws:value"
        data = {}
        data["product"] = product
        return self._parsedresponse(path, argdict, parser, data=data)

    def getBrowse(self, fileId):
        """fileIds is a single file-ID"""
        path = "/getBrowse"
        parser = _parselistofdicts
        argdict = {}
        argdict["containerstr"] = "return"
        argdict["prefix"] = "mws:"
        argdict["listofkeys"] = ["fileID", "product", "description"]
        data = {}
        data["fileId"] = fileId
        return self._parsedresponse(path, argdict, parser, data=data)

    def getCollections(self, product):
        """Available collections for a product"""
        path = "/getCollections"
        parser = _parsekeyvals
        argdict = {}
        argdict["containerstr"] = "mws:Collection"
        argdict["keystr"] = "mws:Name"
        argdict["valstr"] = "mws:Description"
        data = {}
        data["product"] = product
        return self._parsedresponse(path, argdict, parser, data=data)

    def getDataLayers(self, product):
        """Available data layers for a product"""
        path = "/getDataLayers"
        parser = _parsekeyvals
        argdict = {}
        argdict["containerstr"] = "return"
        argdict["keystr"] = "mws:name"
        argdict["valstr"] = "mws:value"
        data = {}
        data["product"] = product
        return self._parsedresponse(path, argdict, parser, data=data)

    def getDateCoverage(self, collection, product):
        """Available dates for a collection/product combination

        TODO: add some result postprocessing - not a good format"""
        path = "/getDateCoverage"
        parser = _parselist
        argdict = {}
        argdict["containerstr"] = "return"
        data = {}
        data["product"] = product
        data["collection"] = collection
        return self._parsedresponse(path, argdict, parser, data=data)

    def getFileOnlineStatuses(self, fileIds):
        """fileIds is a comma-separated list of file-IDs"""
        path = "/getFileOnlineStatuses"
        parser = _parselistofdicts
        argdict = {}
        argdict["containerstr"] = "return"
        argdict["prefix"] = "mws:"
        argdict["listofkeys"] = ["fileID", "archiveAutoDelete", "requireUntil"]
        data = {}
        data["fileIds"] = fileIds
        return self._parsedresponse(path, argdict, parser, data=data)

    def getFileProperties(self, fileIds):
        """fileIds is a comma-separated list of file-IDs"""
        path = "/getFileProperties"
        parser = _parselistofdicts
        argdict = {}
        argdict["containerstr"] = "return"
        argdict["prefix"] = "mws:"
        argdict["listofkeys"] = [
            "fileID",
            "fileName",
            "checksum",
            "fileSizeBytes",
            "fileType",
            "ingestTime",
            "online",
            "startTime",
        ]
        data = {}
        data["fileIds"] = fileIds
        return self._parsedresponse(path, argdict, parser, data=data)

    def getFileUrls(self, fileIds):
        """fileIds is a comma-separated list of file-IDs"""
        path = "/getFileUrls"
        parser = _parselist
        argdict = {}
        argdict["containerstr"] = "return"
        data = {}
        data["fileIds"] = fileIds
        return self._parsedresponse(path, argdict, parser, data=data)

    def getMaxSearchResults(self):
        """Max number of search results that can be returned"""
        path = "/getMaxSearchResults"
        parser = _parselist
        argdict = {}
        argdict["containerstr"] = "ns:return"
        return self._parsedresponse(path, argdict, parser)

    def getOrderStatus(self, orderId):
        """Order status for an order ID"""
        path = "/getOrderStatus"
        parser = _parselist
        argdict = {}
        argdict["containerstr"] = "return"
        data = {}
        data["orderId"] = orderId
        return self._parsedresponse(path, argdict, parser, data=data)

    def getOrderUrl(self, OrderID):
        """Order URL(?) for order ID. TODO: implement"""
        raise NotImplementedError(
            "Method {} not implemented. Probably won't be.".format("getOrderUrl")
        )

    def getPostProcessingTypes(self, products):
        """Products: comma-concatenated string of valid product labels"""
        path = "/getPostProcessingTypes"
        parser = _parselist
        argdict = {}
        argdict["containerstr"] = "return"
        data = {}
        data["products"] = products
        return self._parsedresponse(path, argdict, parser, data=data)

    def listCollections(self):
        """Lists all collections. Deprecated: use getCollections"""
        path = "/listCollections"
        parser = _parsekeyvals
        argdict = {}
        argdict["containerstr"] = "ns:return"
        argdict["keystr"] = "id"
        argdict["valstr"] = "value"
        return self._parsedresponse(path, argdict, parser, unstabletags=True)

    def listMapProjections(self):
        """Lists all available map projections"""
        path = "/listMapProjections"
        parser = _parsekeyvals
        argdict = {}
        argdict["containerstr"] = "ns:return"
        argdict["keystr"] = "name"
        argdict["valstr"] = "value"
        return self._parsedresponse(path, argdict, parser, unstabletags=True)

    def listProductGroups(self, instrument):
        """Lists all available product groups"""
        path = "/listProductGroups"
        parser = _parsekeyvals
        argdict = {}
        argdict["containerstr"] = "return"
        argdict["keystr"] = "mws:name"
        argdict["valstr"] = "mws:value"
        data = {}
        data["instrument"] = instrument
        return self._parsedresponse(path, argdict, parser, data=data)

    def listProducts(self):
        """Lists all available products"""
        path = "/listProducts"
        parser = _parsekeyvals
        argdict = {}
        argdict["containerstr"] = "mws:Product"
        argdict["keystr"] = "mws:Name"
        argdict["valstr"] = "mws:Description"
        return self._parsedresponse(path, argdict, parser)

    def listProductsByInstrument(self, instrument, group=None):
        """Lists all available products for an instrument"""
        path = "/listProductsByInstrument"
        parser = _parselist
        argdict = {}
        argdict["containerstr"] = "return"
        data = {}
        data["instrument"] = instrument
        if group:
            data["group"] = group
        return self._parsedresponse(path, argdict, parser, data=data)

    def listReprojectionParameters(self, reprojectionName):
        """Lists reprojection parameters for a reprojection"""
        path = "/listReprojectionParameters"
        parser = _parselistofdicts
        argdict = {}
        argdict["containerstr"] = "return"
        argdict["prefix"] = "mws:"
        argdict["listofkeys"] = ["name", "description", "units"]
        data = {}
        data["reprojectionName"] = reprojectionName
        return self._parsedresponse(path, argdict, parser, data=data)

    def listSatelliteInstruments(self):
        """Lists all available satellite instruments"""
        path = "/listSatelliteInstruments"
        parser = _parsekeyvals
        argdict = {}
        argdict["containerstr"] = "ns:return"
        argdict["keystr"] = "name"
        argdict["valstr"] = "value"
        return self._parsedresponse(path, argdict, parser, unstabletags=True)

    def orderFiles(
        self,
        email,
        FileIDs,
        reformatType=False,
        doMosaic=False,
        geoSubsetNorth=None,
        geoSubsetSouth=None,
        geoSubsetWest=None,
        geoSubsetEast=None,
        subsetDataLayer=None,
    ):
        """Submits an order"""
        path = "/orderFiles"
        parser = _parselist
        argdict = {}
        argdict["containerstr"] = "return"
        data = {}
        data["email"] = email
        data["fileIds"] = FileIDs
        if reformatType:
            data["reformatType"] = reformatType
        if doMosaic:
            data["doMosaic"] = "true"
        if geoSubsetNorth:
            data["geoSubsetNorth"] = geoSubsetNorth
        if geoSubsetSouth:
            data["geoSubsetSouth"] = geoSubsetSouth
        if geoSubsetWest:
            data["geoSubsetWest"] = geoSubsetWest
        if geoSubsetEast:
            data["geoSubsetEast"] = geoSubsetEast
        if subsetDataLayer:
            data["subsetDataLayer"] = subsetDataLayer

        return self._parsedresponse(path, argdict, parser, data=data)

    def searchForFiles(
        self,
        products,
        startTime,
        endTime,
        north,
        south,
        east,
        west,
        coordsOrTiles="coords",
        dayNightBoth="DNB",
        collection=6,
    ):
        """Submits a search based on product, geography and time"""
        path = "/searchForFiles"
        parser = _parselist
        argdict = {}
        argdict["containerstr"] = "return"
        data = {}
        data["products"] = products
        data["startTime"] = startTime
        data["endTime"] = endTime
        data["north"] = north
        data["south"] = south
        data["east"] = east
        data["west"] = west
        data["coordsOrTiles"] = coordsOrTiles
        if collection:
            data["collection"] = collection
        if dayNightBoth:
            data["dayNightBoth"] = dayNightBoth
        return self._parsedresponse(path, argdict, parser, data=data)

    def searchForFilesByName(self, collection, pattern):
        """Submits a search based on a file name pattern"""
        path = "/searchForFilesByName"
        parser = _parselist
        argdict = {}
        argdict["containerstr"] = "return"
        data = {}
        data["collection"] = collection
        data["pattern"] = pattern
        return self._parsedresponse(path, argdict, parser, data=data)


if __name__ == "__main__":
    a = ModapsClient()
    req = a.listCollections()
    print(req)
