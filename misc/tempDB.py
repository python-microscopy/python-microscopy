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
    cursor.execute ("SELECT time, temp FROM temp WHERE chan=%s AND time > %s AND time < %s", (chan, startTime, endTime))
    res = cursor.fetchall()
    cursor.close ()
    conn.close()

    return numpy.array(res).astype('d').T