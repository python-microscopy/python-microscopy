The Sample Database
*******************

The sample database is a database system which can be used in conjunction with PYMEAcquire to record information about 
sample preparation, labelling, etc for every image that is acquired. The database has a web interface, and is searchable,
allowing data matching various search keys to be easily located.

Structure / Schema
==================

The core unit unit of the database schema is the **Slide**, which represents one stained coverslip (ie all preparation conditions
consistent). The slide is linked to a sample, which was conceived to represent one tissue source (a patient, a rat, a
particular passage of cell culture). Each slide has an associated species, with the **Species** object representing both
species and also strain (e.g. wistar for rats, HeLa, HEK293, etc for cultured cells).

A **Slide** will have one or more **Labelling** s which represent, e.g. the expression of a fluorescent protein,
antibody labelling with a given combination of primary and secondary antibodies, or FISH staining with a specific probe.
Each **Labelling** has a **Dye** associated with it - the long name is what the manufacturer would call it e.g.
'Alexa Fluor 647'. For fluorescent proteins, it should ideally include enough info to uniquely identify it. The short name
is used for internal shorthand and as a key in other parts of our software. It should not include spaces - e.g. 'A647'.
The *spectraDBName* is the name of the dye in the Chroma spectra viewer, to facilitate automatic retrieval of dye spectra.

A **Slide** may have one or more images associated with it. Each image represents one *RAW* data acquisition, and may
have multiple **File** s. One of these files will be the raw data, whilst others could be analysed results.

.. figure:: images/SampleDB_schem_no_tags.png

    Simplified database schema. In addition to the tables depicted, there are also Slide, Image, and File tags which can
    be associated with a given Slide, Image, or File.



Installation
============

These instructions assume you are running an ubuntu linux server with python, mercurial, apache, mysql, and phpmyadmin [#]_. They should also provide a starting point for other systems. A reasonable knowledge of linux and python is assumed. 

Part I - Basic Setup
--------------------

1.  Using ``apt-get`` or ``synaptic`` install ``python-setuptools``, ``python-scipy``, ``python-matplotlib`` and ``python-tables`` from the distribution packages

2.  Get a copy of ``python-microscopy`` by cloning from the repository:
    ::
        hg clone https://code.google.com/p/python-microscopy/
 
3.  Install Django from the distribution package archive - ie ``sudo apt-get install python-django``

4.  Make a new mysql user called ``sample_db`` with password ``PYMEUSER`` (can be modified in ``SampleDB2/settings.py``).

5.  Create a new mysql database called ``sample_db`` and grant the ``sample_db`` user all rights. [#]_

6.  Open a terminal window and change to the ``PYME/SampleDB2/`` directory.

7.  Test the installation so far by entering:
    ::
        python manage.py sql samples
    
    This should show the SQL which will be used to make the tables that `SampleDB` needs, and will only work if Django is installed and can connect to the database. If this fails, chase up the error messages.

8.  Create the tables by entering:
    ::
        python manage.py syncdb 

    This will create all the database tables and prompt you for an admin username and password.

9.  Test the setup by running the development server:
    ::
        python setup.py runserver

    Now direct your browser to ``localhost:8080`` and you should get your first glimpse of the sample database. This will, however, only be visible on the local computer.

Usernames, database names, and passwords can be customized for your site in ``SampleDB2/settings.py`` - for more details see the Django documentation.

.. note:: These instructions follow my re-installation on Ubuntu 14.04 LTS, which ships with Django 1.6. Other versions of Django might not work.

.. [#] phpmyadmin can be substituted for your mysql admin interface of choice
.. [#] steps 2&3 can be combined in phpmyadmin by checking a box during the user creation process

Part II - Getting Apache to serve the SampleDB
----------------------------------------------

.. warning :: In its default state, the PYME SampleDB is not secure. Only use behind a firewall and do so at your own risk/discretion. The version of the Django ``settings.py`` in the python-microscopy repository has ``DEBUG`` set to ``True``, which is a known security risk. In a controlled environment, this risk is probably acceptable in return for easier troubleshooting, but you have been warned! 



1.  Create a directory ``/var/www/SampleDB/static`` for the static files (if you want to host the files from another directory, you will need to change ``STATIC_ROOT`` in ``settings.py`` and the apache .conf file detailed in step 3). 

2.  Install the static files by calling:
    ::
        sudo python manage.py collectstatic  

3.  Create a new file in ``/etc/apache2/conf-available`` called ``SampleDB.conf`` with the following contents (alter the paths to reflect where you have extracted python-microscopy):
    ::
        WSGIScriptAlias / /home/david/python-microscopy/PYME/SampleDB2/SampleDB2/wsgi.py
        WSGIPythonPath /home/david/python-microscopy/PYME/SampleDB2/

        <Directory /home/david/python-microscopy/PYME/SampleDB2/SampleDB2/>
        <Files wsgi.py>
        Require all granted
        </Files>
        </Directory>

        Alias /media/ /var/www/SampleDB/static/
        <Directory /var/www/SampleDB/static/>
        Order deny,allow
        Allow from all
        </Directory>

4.  Activate the newly created ``SampleDB.conf`` by calling:
    ::
        sudo a2enconf SampleDB
        sudo service apache2 reload

5.  Verify that you can now see the server from another machine.

6.  **[Optional but reccomended]** Lock the server down. Edit ``settings.py`` to add your machine name to ``ALLOWED_HOSTS`` and then set ``DEBUG`` to ``False``. Restart apache with ``sudo service apache2 reload`` to make the changes take effect.
    
    .. warning :: This alone is not enough to make SampleDB secure. You would also want to look at changing the database passwords and the ``SECRET_KEY`` in ``settings.py``, as well as potentially restricting access to MySQL to the local machine. Some items are stored in the database as pickles, which means that, although difficult to exploit, a database breach theoretically has the capablilty to allow remote code execution.
        
Part III - Letting other machines know where to find the SampleDB
------------------------------------------------------------------

Letting other machines find the sample database is as simple as setting an environment variable: ``PYME_DATABASE_HOST`` to the hostname or IP address of the server.

