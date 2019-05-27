import gmplot
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog
from gmplot import gmplot

import pprint


def get_links_with_coord(xmlData):

    source_target = []

#itterate thru the link
    for link in xmlData[0][1]:
        #get the link source and target name
        linkSource = link.find('source').text
        linkTarget = link.find('target').text

        #empty list to store x and y coordinates of the source and target
        x_source_coor = []
        y_source_coor = []

        x_target_coor = []
        y_target_coor = []

        #make a new list of (source-->target) with coordinates in the format [ [source,target],[x_source],[ x_target], [y_source],[y_target] ]
        source_target.append([[linkSource,linkTarget],x_source_coor,x_target_coor,y_source_coor,y_target_coor])

        #iterate thru the nodes
        for node in xmlData[0][0]:
            #get the node name
            nodeName = node.attrib.get('id')

            # getting the coordinate
            coordObj = node.find('./coordinates')
            xCoord = coordObj.find('x').text
            yCoord = coordObj.find('y').text

            #compare between the node name and link source
            if nodeName == linkSource:
                source_x = (float(xCoord))
                x_source_coor.append(source_x)

                source_y = (float(yCoord))
                y_source_coor.append(source_y)

            if nodeName == linkTarget:
                target_x = (float(xCoord))
                x_target_coor.append(target_x)
                target_y = (float(yCoord))
                y_target_coor.append(target_y)

    return source_target

#create the map in html+javascript, you need API key
def create_map(xmlData):
    links_list = get_links_with_coord(xmlData)

    # set the map point based on first link beware the latitude and longitude value given
    gmap3 = gmplot.GoogleMapPlotter(links_list[0][3][0], links_list[0][1][0], 13)
    # separated the coordinates
    # longitude_list, latitude_list = zip(*links_with_coord(xmlData))

    # rearrange the elements in links_list to latitude_list and longitude_list with format [x_source,x_target] and [y_source,y_target]
    # use the list to plot everytime
    for links in links_list:
        latitude_list = links[1] + links[2]
        longitude_list = links[3] + links[4]

        gmap3.scatter(longitude_list, latitude_list, '# FF0000',
                      size=40, marker=False)

        # Plot method Draw a line in
        # between given coordinates
        gmap3.plot(longitude_list, latitude_list,
                   'cornflowerblue', edge_width=2.5)

    gmap3.draw("Map.html")
    # use your own API key, gedit from Google Map API
    insertapikey("my_map_test4.html", 'Your Api Key')

#install BeautifulSoup
from bs4 import BeautifulSoup
# this function is used to embedded your api key in the html
def insertapikey(fname, apikey):
    """put the google api key in a html file"""
    def putkey(htmltxt, apikey, apistring=None):
        """put the apikey in the htmltxt and return soup"""
        if not apistring:
            apistring = "https://maps.googleapis.com/maps/api/js?key=%s&callback=initMap"
        soup = BeautifulSoup(htmltxt, 'html.parser')
        body = soup.body
        src = apistring % (apikey, )
        tscript = soup.new_tag("script", src=src, async="defer")
        body.insert(-1, tscript)
        return soup
    htmltxt = open(fname, 'r').read()
    soup = putkey(htmltxt, apikey)
    newtxt = soup.prettify()
    open(fname, 'w').write(newtxt)

def main():
    root = tk.Tk()
    #pp = pprint.PrettyPrinter(indent=4)

    root.withdraw()

    xmlObject = filedialog.askopenfilename()
    xmlData = ET.parse(xmlObject).getroot()

    create_map(xmlData)

main()
