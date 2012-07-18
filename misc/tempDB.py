#!/usr/bin/python

###############
# tempDB.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################
import MySQLdb
import numpy

def getConn():
    return MySQLdb.connect (host = "lmsrv1",
                           user = "PYMEUSER",
                           passwd = "PYME",
                           db = "temperature")

def addEntry(time, temp, chan=1):
    conn = getConn()
    cursor = conn.cursor ()
    cursor.execute ("INSERT INTO temp (time, temp, chan) VALUES (%s,%s,%s)", (time, temp, chan))
    cursor.close ()
    conn.close()

def addEntries(vals, chan=1):
    conn = getConn()
    cursor = conn.cursor ()
    for time, temp in vals:
        #print str(time), str(temp - 273.15)
        cursor.execute ("INSERT INTO temp (time, temp, chan) VALUES (%s,%s,%s)", (time, temp - 273.15, chan))
    cursor.close ()
    conn.close()

def getEntries(startTime=0, endTime=1e20, chan=1):
    conn = getConn()
    cursor = conn.cursor ()
    cursor.execute ("SELECT time, temp FROM temp WHERE chan=%s AND time > %s AND time < %s ORDER BY time", (chan, startTime, endTime))
    res = cursor.fetchall()
    cursor.close ()
    conn.close()

    return numpy.array(res).astype('d').T